"""
Generates a cited answer from retrieved chunks using an LLM on Groq.

Prompt design is driven by a measured property of this benchmark: about half of
FinanceBench answers do not appear anywhere in their own evidence, because the
expected value must be calculated (ratios, multi-year averages, year-over-year
deltas). A naive "answer only from the context" instruction would fail those by
construction, so the prompt explicitly requires extraction THEN arithmetic.

Default model is openai/gpt-oss-120b: the strongest model on Groq's free tier
for this key, and a reasoning model, which matters for the arithmetic.
Free-tier limits (read from response headers): 1000 requests/day and 8000
tokens/minute -- the per-minute token budget is the binding constraint, so the
caller paces requests and this class waits out short 429s.
"""
import os
import re
import time

from dotenv import load_dotenv
from groq import APIStatusError, Groq, RateLimitError

load_dotenv()

DEFAULT_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
# Reasoning tokens count against the 8000 tokens/minute budget; "medium" is
# enough for multi-step arithmetic without exhausting a minute on one question.
REASONING_EFFORT = os.getenv("GROQ_REASONING_EFFORT", "medium")
MAX_COMPLETION_TOKENS = 3000
# A 429 asking us to wait longer than this is a spent daily quota, not a
# per-minute window: stop the run (answers are cached) instead of sleeping for hours.
MAX_RATE_LIMIT_WAIT = 120
RETRY_DELAYS = (5, 15, 30, 60)  # for transient 5xx errors

PROMPT_TEMPLATE = """You are a financial analyst. Answer the question using the SEC filing excerpts below.

RULES:
1. Use ONLY these excerpts. Do not use outside knowledge about the company.
2. Many questions require CALCULATION -- ratios, multi-year averages, year-over-year
   changes. The final number often does NOT appear in the text. Extract the component
   figures from the excerpts, then compute it. Show your arithmetic.
3. Mind the units. These filings usually report in millions. State the unit.
4. Mind the fiscal year. Excerpts often show several years side by side in one table;
   make sure you read the column for the year the question asks about.
5. If the excerpts genuinely lack the figures needed, write exactly: NOT FOUND
   Do not guess a plausible number.

Reply in EXACTLY this format, with no extra commentary:
REASONING: <figures you used and the arithmetic you performed>
FINAL ANSWER: <the value with its unit, or NOT FOUND>
CITATIONS: <page numbers you used>

EXCERPTS:
{context}

QUESTION: {question}
"""

FINAL_ANSWER_RE = re.compile(r"FINAL ANSWER:\s*(.+?)(?:\n|$)", re.IGNORECASE)
CITATIONS_RE = re.compile(r"CITATIONS:\s*(.+?)(?:\n|$)", re.IGNORECASE)


class DailyLimitReached(Exception):
    """The provider asked us to wait longer than a per-minute window: quota is spent."""


class AnswerGenerator:
    def __init__(self, model: str | None = None):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise SystemExit("GROQ_API_KEY not set. Add it to .env")
        # SDK-level retries off: rate-limit waits are decided here, where a long
        # wait can be recognised as a spent daily quota.
        self.client = Groq(api_key=api_key, max_retries=0)
        self.model = model or DEFAULT_MODEL

    @staticmethod
    def build_context(chunks: list[dict]) -> str:
        """One labelled block per chunk, so citations can reference real pages."""
        blocks = []
        for chunk in chunks:
            pages = (
                f"{chunk['page_start']}"
                if chunk["page_start"] == chunk["page_end"]
                else f"{chunk['page_start']}-{chunk['page_end']}"
            )
            blocks.append(
                f"[Document: {chunk['doc_name']} | Page: {pages}]\n{chunk['text']}"
            )
        return "\n\n---\n\n".join(blocks)

    def generate(self, question: str, chunks: list[dict]) -> dict:
        prompt = PROMPT_TEMPLATE.format(
            context=self.build_context(chunks), question=question
        )
        response, latency, error = self._call_with_retry(prompt)

        if response is None:
            return {
                "answer": "ERROR",
                "reasoning": "",
                "citations": "",
                "raw": "",
                "latency": latency,
                "input_tokens": 0,
                "output_tokens": 0,
                "error": error,
            }

        text = response.choices[0].message.content or ""
        usage = response.usage
        return {
            "answer": self._extract(FINAL_ANSWER_RE, text) or text.strip()[:200],
            "reasoning": self._extract_reasoning(text),
            "citations": self._extract(CITATIONS_RE, text),
            "raw": text,
            "latency": latency,
            "input_tokens": usage.prompt_tokens if usage else 0,
            "output_tokens": usage.completion_tokens if usage else 0,
            "error": None,
        }

    def _call_with_retry(self, prompt: str):
        start = time.time()
        last_error = None
        transient_attempts = 0
        while True:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    reasoning_effort=REASONING_EFFORT,
                    max_completion_tokens=MAX_COMPLETION_TOKENS,
                )
                return response, time.time() - start, None
            except RateLimitError as exc:
                wait = self._retry_after(exc)
                if wait > MAX_RATE_LIMIT_WAIT:
                    raise DailyLimitReached(
                        f"Groq asked to wait {wait:.0f}s -- daily quota reached"
                    ) from exc
                print(f"    [rate limited; waiting {wait:.0f}s]", flush=True)
                time.sleep(wait)
            except APIStatusError as exc:
                last_error = exc
                if exc.status_code < 500 or transient_attempts >= len(RETRY_DELAYS):
                    break
                time.sleep(RETRY_DELAYS[transient_attempts])
                transient_attempts += 1
            except Exception as exc:  # noqa: BLE001 -- network errors etc.
                last_error = exc
                if transient_attempts >= len(RETRY_DELAYS):
                    break
                time.sleep(RETRY_DELAYS[transient_attempts])
                transient_attempts += 1

        error = f"{type(last_error).__name__}: {str(last_error)[:150]}"
        print(f"    [generation failed: {error}]")
        return None, time.time() - start, error

    @staticmethod
    def _retry_after(exc: RateLimitError) -> float:
        """Seconds Groq asks us to wait; falls back to a minute if unstated."""
        headers = exc.response.headers if exc.response is not None else {}
        try:
            return float(headers.get("retry-after", 60))
        except (TypeError, ValueError):
            return 60.0

    @staticmethod
    def _extract(pattern: re.Pattern, text: str) -> str:
        match = pattern.search(text)
        return match.group(1).strip() if match else ""

    @staticmethod
    def _extract_reasoning(text: str) -> str:
        match = re.search(
            r"REASONING:\s*(.*?)(?=FINAL ANSWER:|$)", text, re.IGNORECASE | re.DOTALL
        )
        return match.group(1).strip() if match else ""
