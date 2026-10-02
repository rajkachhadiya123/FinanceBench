"""
Generates a cited answer from retrieved chunks using Gemini.

Prompt design is driven by a measured property of this benchmark: 38% of
FinanceBench answers do not appear anywhere in their own evidence, because the
expected value must be calculated (ratios, multi-year averages, year-over-year
deltas). A naive "answer only from the context" instruction would fail those by
construction, so the prompt explicitly requires extraction THEN arithmetic.
"""
import os
import re
import time

from dotenv import load_dotenv
from google import genai

load_dotenv()

DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
FALLBACK_MODEL = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-3.7-flash")
# 503 "high demand" is transient on the free tier. The earlier delays
# (2,5,12,25) exhausted after 44s and lost 20% of answers, so they are longer
# now and the last attempts also switch model.
# Tail delays are ~60s because per-minute rate windows need that long to reset.
RETRY_DELAYS = (2, 5, 15, 30, 60, 65)
# Attempt index at which to stop asking the congested primary model.
SWITCH_MODEL_AFTER = 3

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


class AnswerGenerator:
    """
    Holds one or more API keys and rotates to the next when a key's quota is
    exhausted (429). That is a different failure from 503 "high demand", which
    is transient capacity on the same key and is handled by waiting instead.
    """

    def __init__(self, model: str | None = None):
        keys = [
            key for key in (os.getenv("GEMINI_API_KEY"), os.getenv("GEMINI_API_KEY_2"))
            if key
        ]
        if not keys:
            raise SystemExit("GEMINI_API_KEY not set. Add it to .env")
        self.keys = keys
        self.key_index = 0
        self.client = genai.Client(api_key=keys[0])
        self.model = model or DEFAULT_MODEL
        self.quota_switches = 0

    def _next_key(self) -> None:
        """
        Round-robin to the next key. Deliberately wraps rather than dead-ending:
        a 429 on the free tier is usually a PER-MINUTE limit, not a spent daily
        quota, so coming back to an earlier key after the retry delay works.
        Treating the last key's 429 as terminal previously failed 26 of 30
        questions in a single run.
        """
        if len(self.keys) > 1:
            self.key_index = (self.key_index + 1) % len(self.keys)
            self.client = genai.Client(api_key=self.keys[self.key_index])
            self.quota_switches += 1

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
        response, latency = self._call_with_retry(prompt)

        if response is None:
            return {
                "answer": "ERROR",
                "reasoning": "",
                "citations": "",
                "raw": "",
                "latency": latency,
                "input_tokens": 0,
                "output_tokens": 0,
                "error": "all retries exhausted",
            }

        text = response.text or ""
        usage = response.usage_metadata
        return {
            "answer": self._extract(FINAL_ANSWER_RE, text) or text.strip()[:200],
            "reasoning": self._extract_reasoning(text),
            "citations": self._extract(CITATIONS_RE, text),
            "raw": text,
            "latency": latency,
            "input_tokens": usage.prompt_token_count or 0,
            "output_tokens": usage.candidates_token_count or 0,
            "error": None,
        }

    def _call_with_retry(self, prompt: str):
        start = time.time()
        last_error = None
        for attempt, delay in enumerate((0, *RETRY_DELAYS)):
            if delay:
                time.sleep(delay)

            # After a few failures the primary model is likely congested, so
            # try the fallback rather than waiting on the same queue.
            model = self.model if attempt < SWITCH_MODEL_AFTER else FALLBACK_MODEL
            try:
                response = self.client.models.generate_content(
                    model=model, contents=prompt
                )
                return response, time.time() - start
            except Exception as exc:  # noqa: BLE001 -- provider raises varied types
                last_error = exc
                if self._is_quota_error(exc):
                    self._next_key()  # try the other key; delays let limits reset
                    continue
                if not self._is_retryable(exc):
                    break

        print(f"    [generation failed: {type(last_error).__name__}: {str(last_error)[:90]}]")
        return None, time.time() - start

    @staticmethod
    def _is_quota_error(exc: Exception) -> bool:
        """Quota exhausted -- a different key may still work."""
        if getattr(exc, "code", None) == 429:
            return True
        return "RESOURCE_EXHAUSTED" in str(exc) or "429" in str(exc)

    @staticmethod
    def _is_retryable(exc: Exception) -> bool:
        """Transient capacity -- the same key will likely work if we wait."""
        if getattr(exc, "code", None) in (500, 502, 503, 504):
            return True
        return any(s in str(exc) for s in ("503", "UNAVAILABLE", "500", "internal"))

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
