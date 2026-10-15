"""
Generates a cited answer from retrieved chunks using an LLM on Cerebras.

Prompt design is driven by a measured property of this benchmark: about half of
FinanceBench answers do not appear anywhere in their own evidence, because the
expected value must be calculated (ratios, multi-year averages, year-over-year
deltas). A naive "answer only from the context" instruction would fail those by
construction, so the prompt explicitly requires extraction THEN arithmetic.

Model: gpt-oss-120b on Cerebras -- the same reasoning model previously used on
Groq, so results stay comparable. Cerebras serves an OpenAI-compatible API, so
it is called with plain HTTP (no extra SDK).
Limits for this key (read from response headers): 5 requests/min, 150/hour,
2,400/day; 30,000 tokens/min, 1,000,000/day. Requests per minute is the binding
constraint; the caller paces requests and this class waits out 429s.

History: V2 used Groq's free tier (openai/gpt-oss-120b, 8K tokens/min), which
needed two keys and several days per run. The Groq setup is kept commented out
below.
"""
import os
import re
import time

import httpx
from dotenv import load_dotenv

# --- Groq (retired; daily free-tier quota too small) -------------------------
# from groq import APIStatusError, Groq, RateLimitError
# DEFAULT_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
# REASONING_EFFORT = os.getenv("GROQ_REASONING_EFFORT", "medium")
# ------------------------------------------------------------------------------

load_dotenv()

API_URL = "https://api.cerebras.ai/v1/chat/completions"
DEFAULT_MODEL = os.getenv("CEREBRAS_MODEL", "gpt-oss-120b")
# "medium" is enough for multi-step arithmetic without wasting the token budget.
# "high" was tested with prompt v4 on the 31 failed questions: no calculation
# fixed, and 5 answers came back empty because hidden reasoning used all 6,000
# completion tokens. Reverted to medium / 3,000 with prompt v2.
REASONING_EFFORT = os.getenv("CEREBRAS_REASONING_EFFORT", "medium")
MAX_COMPLETION_TOKENS = 3000
# A 429 asking us to wait longer than this is a spent daily quota: stop the run
# (answers are cached) instead of sleeping for hours. An exhausted hourly
# window (150 requests) can ask for up to an hour, which we wait out.
MAX_RATE_LIMIT_WAIT = 3700
RETRY_DELAYS = (5, 15, 30, 60)  # for transient 5xx / network errors

# v2: general rules derived from failure patterns in the v1 run (formula not
# applied as defined, sign errors, conclusions without supporting figures,
# conclusions contradicting the computed numbers, wrong filing/period, generic
# "what drove" answers, truncated lists). No rule names a company, question or
# expected value. v1 is kept below for comparison.
# A v3 (v2 + "use consolidated statements / never assume zero / no metric
# substitution") was tested and removed: 0 of 16 wrong answers fixed, 1 correct
# answer lost. The model cannot tell parent-company-only tables apart because
# chunks do not carry their statement headings -- a chunking problem.
# v4: changes the answer FORMAT rather than adding rules (v3 showed more rules
# alone do nothing). Analysis of the headed run: in 7 of 11 wrong calculations
# every input was in the chunks; the model took a figure from the wrong row or
# column, dropped a sign, or skipped the final step. v4 makes it copy each
# input's row label, period and source, write the formula with the numbers, and
# run explicit checks before answering. It also explains the chunk headers.
# No company, question or expected value is named; example numbers are invented.
# Result (with reasoning effort "high", on the 31 questions the headed v2 run
# failed): 4 fixed -- all non-calculation, 3 of them earlier run-to-run flips --
# 0 of 18 calculation failures fixed, 5 empty answers. Not adopted; v2 stays
# active (119/150 = 79.3% with headed retrieval). v4 is kept below for reference.
PROMPT_VERSION = "v2"

PROMPT_TEMPLATE_V4 = """You are a financial analyst. Answer the question using ONLY the SEC filing excerpts below. Do not use outside knowledge about the company.

Each excerpt begins with a label: [COMPANY | FILING TYPE | PERIOD | page N | section]. The section names the statement or note the text comes from (for example a consolidated balance sheet, a cash flow statement, a numbered note, or a parent-company-only schedule).

STEP 1 - UNDERSTAND THE QUESTION
- Decide the answer type: a number, a yes/no judgement, a comparison, a list, or an explanation.
- Identify the exact metric, company and fiscal period(s) asked for. If the question defines a metric or formula, apply that definition exactly, even if a more common definition exists.

STEP 2 - FIND EACH INPUT (for every figure you use)
- Find the table row whose label matches the item you need, and copy that row label exactly as printed.
- Take the value from the column for the requested period. Filings show several periods side by side: check the column header (which year or quarter; "three months ended" versus "year ended").
- Use the consolidated statements unless the question asks for a segment, a quarter or the parent company alone.
- Use the metric the question names. If it is not reported directly, build it from its components. Do not replace it with a similar but different metric (for example an adjusted figure instead of an unadjusted one, or a broader or narrower earnings measure).

STEP 3 - CALCULATE
- Write the formula with the actual numbers substituted, then the result.
- Keep signs as reported: losses, outflows and figures shown in parentheses are negative, and a ratio involving a negative figure keeps its sign.
- A percentage of a total is part / total x 100. Growth or change is (new - old) / old.
- Round as the question instructs.

STEP 4 - CHECK BEFORE ANSWERING
- Is every input from the right company, statement and period?
- Is the denominator the item the formula calls for?
- Is the size plausible? A part cannot exceed its total, a quarter cannot exceed its full year, and a value far outside the normal range for that metric usually means a wrong input. If a check fails, find the correct input and recalculate.
- For yes/no and judgement questions, does the conclusion follow from the figures?

STEP 5 - NON-NUMERIC QUESTIONS
- For yes/no questions, begin the final answer with Yes or No, then give the supporting fact.
- For "what drove" or "why" questions, name the specific factors the excerpts state, not generic causes.
- If the question says a metric may not be useful for this kind of company, decide that first from the nature of the business. If it is not useful, say so and explain why instead of calculating it.
- Only if the excerpts truly lack the needed information, answer exactly: NOT FOUND. Do not guess.

Reply in EXACTLY this format:
INPUTS: <one line per figure: row label as printed | value | unit | period | document, page. Write "none" if no figures are needed.>
FORMULA: <the calculation with numbers substituted, e.g. (1,250 + 300) / 9,800 * 100. Write "none" if there is no calculation.>
REASONING: <brief explanation, including the checks from step 4>
FINAL ANSWER: <a complete, self-contained answer: the conclusion or value first, then the key supporting figure(s) with units. A list may span several lines.>
CITATIONS: <document and page numbers used>

EXCERPTS:
{context}

QUESTION: {question}
"""

PROMPT_TEMPLATE = """You are a financial analyst. Answer the question using ONLY the SEC filing excerpts below. Do not use outside knowledge about the company.

HOW TO ANSWER:
1. Decide what kind of answer is needed: a number, a yes/no judgement, a comparison, a list, or an explanation.
2. Check each excerpt's [Document | Page] label. Use excerpts from the company and the fiscal period the question asks about. If excerpts from different filings or periods disagree, use the one matching the period in the question.
3. If the question defines a metric or formula, apply that definition exactly as written, using the line items it names. Do not substitute a different or more common version of the metric.
4. For calculations: list every input figure with its period and unit, then compute step by step. Keep signs as reported (a loss or negative value stays negative). Growth or change = (new - old) / old. Round as the question instructs.
5. Read the right column: filings show several years side by side. State units, and answer in the unit the question requests.
6. For judgement questions (is it healthy, consistent, improving, capital-intensive, etc.): compute or quote the relevant figure first, then make sure the conclusion follows from that figure. Give both the conclusion and the figure.
7. For "what drove" or "why" questions: name the specific factors the excerpts state, not generic causes.
8. If the question says a metric may not be relevant for this kind of company, decide whether it is from the excerpts, and if it is not, say so and explain why.
9. Only if the excerpts truly lack the needed information, answer exactly: NOT FOUND. Do not guess.

Reply in EXACTLY this format:
REASONING: <the figures you used, with document/page, and each calculation step>
FINAL ANSWER: <a complete, self-contained answer: the conclusion or value first, then the key supporting figure(s) with units. A list may span several lines.>
CITATIONS: <document and page numbers used>

EXCERPTS:
{context}

QUESTION: {question}
"""

PROMPT_TEMPLATE_V1 = """You are a financial analyst. Answer the question using the SEC filing excerpts below.

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

# Everything between FINAL ANSWER: and CITATIONS: -- v1 kept only the first
# line, which truncated list answers.
FINAL_ANSWER_RE = re.compile(r"FINAL ANSWER:\s*(.+?)(?=\n\s*CITATIONS:|\Z)",
                             re.IGNORECASE | re.DOTALL)
CITATIONS_RE = re.compile(r"CITATIONS:\s*(.+?)(?:\n|$)", re.IGNORECASE)
FORMULA_RE = re.compile(r"FORMULA:\s*(.+?)(?=\n\s*REASONING:|\Z)", re.IGNORECASE | re.DOTALL)
NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def check_formula(formula: str, answer: str) -> str:
    """
    Recompute the model's FORMULA line and compare it with the first number in
    its FINAL ANSWER: "match", "mismatch" or "n/a" (no formula, or not plain
    arithmetic). Diagnostic only -- the answer is never changed.
    """
    expr = (formula.split("=")[0].replace(",", "").replace("×", "*").replace("÷", "/")
            .replace("−", "-").replace("–", "-").replace("$", "").replace("%", ""))
    expr = re.sub(r"(?<=[\d)])\s*x\s*(?=[\d(])", "*", expr)
    if not expr.strip() or not re.fullmatch(r"[\d\s.+\-*/()]+", expr):
        return "n/a"
    try:
        value = float(eval(expr, {"__builtins__": {}}, {}))  # digits/operators only, checked above
    except (SyntaxError, ZeroDivisionError, TypeError, ValueError):
        return "n/a"
    numbers = NUMBER_RE.findall(answer.replace("‑", "-"))
    if not numbers:
        return "n/a"
    stated = float(numbers[0].replace(",", ""))
    # 1% relative, plus 0.05 absolute so answers rounded to one decimal still match
    close = lambda a, b: abs(a - b) <= 0.01 * max(abs(a), abs(b)) + 0.05  # noqa: E731
    return "match" if any(close(stated, v) for v in (value, value * 100, value / 100)) else "mismatch"


class DailyLimitReached(Exception):
    """The provider asked us to wait longer than a per-minute window: quota is spent."""


class AnswerGenerator:
    def __init__(self, model: str | None = None):
        api_key = os.getenv("CEREBRAS_API_KEY")
        if not api_key:
            raise SystemExit("CEREBRAS_API_KEY not set. Add it to .env")
        self.client = httpx.Client(headers={"Authorization": f"Bearer {api_key}"},
                                   timeout=120)
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
                "formula": "",
                "calc_check": "n/a",
                "raw": "",
                "latency": latency,
                "input_tokens": 0,
                "output_tokens": 0,
                "error": error,
            }

        message = response["choices"][0]["message"]
        text = message.get("content") or ""
        usage = response.get("usage") or {}
        answer = self._extract(FINAL_ANSWER_RE, text) or text.strip()[:200]
        formula = self._extract(FORMULA_RE, text)
        return {
            "answer": answer,
            "reasoning": self._extract_reasoning(text),
            "citations": self._extract(CITATIONS_RE, text),
            "formula": formula,
            "calc_check": check_formula(formula, answer),
            "raw": text,
            "latency": latency,
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "error": None,
        }

    def _call_with_retry(self, prompt: str):
        start = time.time()
        last_error = None
        transient_attempts = 0
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "reasoning_effort": REASONING_EFFORT,
            "max_completion_tokens": MAX_COMPLETION_TOKENS,
        }
        while True:
            try:
                response = self.client.post(API_URL, json=payload)
            except httpx.HTTPError as exc:  # network errors
                last_error = exc
                if transient_attempts >= len(RETRY_DELAYS):
                    break
                time.sleep(RETRY_DELAYS[transient_attempts])
                transient_attempts += 1
                continue

            if response.status_code == 200:
                return response.json(), time.time() - start, None
            if response.status_code == 429:
                wait = self._retry_after(response)
                if wait > MAX_RATE_LIMIT_WAIT:
                    raise DailyLimitReached(
                        f"Cerebras asked to wait {wait:.0f}s -- daily quota reached")
                print(f"    [rate limited; waiting {wait:.0f}s]", flush=True)
                time.sleep(wait)
                continue
            last_error = RuntimeError(f"HTTP {response.status_code}: {response.text[:150]}")
            if response.status_code < 500 or transient_attempts >= len(RETRY_DELAYS):
                break
            time.sleep(RETRY_DELAYS[transient_attempts])
            transient_attempts += 1

        error = f"{type(last_error).__name__}: {str(last_error)[:150]}"
        print(f"    [generation failed: {error}]")
        return None, time.time() - start, error

    @staticmethod
    def _retry_after(response: httpx.Response) -> float:
        """Seconds the provider asks us to wait; falls back to a minute if unstated."""
        for header in ("retry-after", "x-ratelimit-reset-requests-minute"):
            try:
                return float(response.headers[header]) + 1
            except (KeyError, TypeError, ValueError):
                continue
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
