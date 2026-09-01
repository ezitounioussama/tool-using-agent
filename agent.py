"""A tool-using agent with fallback logic.

Route for every question, in order:

  1. score the tools; the best score below THRESHOLD means no tool fits
  2. run the winning tool under a deadline
  3. tool raised, hung, or returned nothing usable -> answer with the LLM
  4. no tool fitted in the first place -> answer with the LLM directly

Steps 3 and 4 both end at the model, but they are different events and the log
keeps them apart: 3 is a fallback (a tool was tried and failed), 4 is a direct
answer (no tool was ever appropriate). Collapsing them would hide tool
breakage, which is exactly what you want to see in production.
"""

import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout

import llm
from tools import TOOLS, ToolError, ToolTimeout

# Below this score, no tool is considered a fit and the model answers directly.
SCORE_THRESHOLD = 0.5

# A prose tool answer shorter than this, or less confident than this, is treated
# as too thin to serve and falls back to the model. Results marked `atomic`
# (a calculator's "96") are exempt -- see the note in tools.ToolResult.
MIN_PROSE_CHARS = 80
MIN_CONFIDENCE = 0.5

DEFAULT_TOOL_TIMEOUT = 5.0

SYSTEM_PROMPT = (
    "You are a concise assistant. Answer in at most three sentences. "
    "If you are not sure of a fact, say so rather than inventing it. "
    "Never claim to have used a tool or looked anything up."
)

ROUTE_TOOL = "TOOL"
ROUTE_FALLBACK = "FALLBACK"
ROUTE_DIRECT = "DIRECT"
ROUTE_UNAVAILABLE = "UNAVAILABLE"


class Answer:
    """One answered question, and the record of how it got answered."""

    def __init__(self, query, text, route, source, reason="", elapsed=0.0, confidence=None):
        self.query = query
        self.text = text
        self.route = route          # TOOL | FALLBACK | DIRECT | UNAVAILABLE
        self.source = source        # tool name, or "LLM"
        self.reason = reason        # why this route was taken
        self.elapsed = elapsed
        self.confidence = confidence

    @property
    def label(self):
        return f"{self.route}/{self.source}"

    def __repr__(self):
        return f"Answer({self.label}, {self.text[:40]!r})"


class Agent:
    def __init__(self, tools=None, generate=None, verbose=True):
        self.tools = list(tools if tools is not None else TOOLS)
        # Injectable so the tests can run without a model.
        self.generate = generate or llm.generate
        self.verbose = verbose
        self.log = []

    # ---------------------------------------------------------------- routing

    def choose_tool(self, query):
        """Return (tool, score) for the best-fitting tool, or (None, score)."""
        scored = [(tool, tool.score(query)) for tool in self.tools]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        best_tool, best_score = scored[0]
        if best_score < SCORE_THRESHOLD:
            return None, best_score
        return best_tool, best_score

    # ------------------------------------------------------------------ tools

    def _run_tool(self, tool, query):
        """Run a tool under its own deadline.

        The deadline is enforced here rather than trusted to the tool, because a
        tool that hangs is the failure the fallback exists for. The worker
        thread is left to finish on its own (daemon), so a stuck tool cannot
        block the answer.
        """
        timeout = getattr(tool, "timeout_seconds", DEFAULT_TOOL_TIMEOUT)
        # Deliberately not `with ThreadPoolExecutor(...)`: leaving the block
        # calls shutdown(wait=True), which joins the hung thread and blocks for
        # its full duration -- the timeout would fire and the agent would still
        # wait. Measured: 2200 ms with the context manager, 2000 ms without.
        pool = ThreadPoolExecutor(max_workers=1)
        future = pool.submit(tool.run, query)
        try:
            result = future.result(timeout=timeout)
        except FutureTimeout as error:
            pool.shutdown(wait=False, cancel_futures=True)
            raise ToolTimeout(f"{tool.name} exceeded its {timeout}s deadline") from error
        pool.shutdown(wait=False)
        return result

    # ------------------------------------------------------------------ answer

    def answer(self, query):
        started = time.perf_counter()
        tool, score = self.choose_tool(query)

        if tool is None:
            reason = f"no tool scored above {SCORE_THRESHOLD} (best {score:.2f})"
            return self._record(self._direct(query, reason, started))

        try:
            result = self._run_tool(tool, query)
        except (ToolError, ToolTimeout) as error:
            kind = "timed out" if isinstance(error, ToolTimeout) else "failed"
            return self._record(
                self._fallback(query, f"{tool.name} {kind}: {error}", started, tool)
            )
        except Exception as error:  # noqa: BLE001 - a tool must never crash the agent
            return self._record(
                self._fallback(
                    query,
                    f"{tool.name} raised an unexpected {type(error).__name__}: {error}",
                    started,
                    tool,
                )
            )

        thin = (not result.atomic) and len(result.text.strip()) < MIN_PROSE_CHARS
        unsure = result.confidence < MIN_CONFIDENCE
        if thin or unsure:
            why = "answer too short" if thin else f"confidence {result.confidence:.2f}"
            return self._record(
                self._fallback(
                    query,
                    f"{tool.name} returned a weak answer ({why})",
                    started,
                    tool,
                    tool_text=result.text,
                )
            )

        return self._record(
            Answer(
                query,
                result.text,
                ROUTE_TOOL,
                tool.name,
                reason=result.detail or f"score {score:.2f}",
                elapsed=time.perf_counter() - started,
                confidence=result.confidence,
            )
        )

    # --------------------------------------------------------------- LLM paths

    def _ask_model(self, prompt):
        return self.generate(prompt, system=SYSTEM_PROMPT)

    def _direct(self, query, reason, started):
        try:
            text = self._ask_model(query)
        except llm.LLMError as error:
            return Answer(
                query,
                "I could not answer that: no tool applied and the language model is unavailable.",
                ROUTE_UNAVAILABLE,
                "none",
                reason=f"{reason}; LLM also failed: {error}",
                elapsed=time.perf_counter() - started,
            )
        return Answer(
            query, text, ROUTE_DIRECT, "LLM", reason=reason, elapsed=time.perf_counter() - started
        )

    def _fallback(self, query, reason, started, tool, tool_text=None):
        """Answer with the model after a tool did not deliver.

        The prompt says a tool was unavailable but never passes the tool's weak
        output as if it were fact -- a thin or unreliable answer is exactly what
        we are trying not to repeat.
        """
        prompt = (
            f"A tool named {tool.name} was unavailable for this question, so answer it "
            f"yourself from what you know.\n\nQuestion: {query}"
        )
        if tool_text:
            prompt += (
                f"\n\nThe tool returned only this fragment, which may be incomplete: "
                f"{tool_text!r}. Use it only if it is consistent with what you know."
            )

        try:
            text = self._ask_model(prompt)
        except llm.LLMError as error:
            return Answer(
                query,
                f"I could not answer that: {tool.name} failed and the language model is "
                "also unavailable.",
                ROUTE_UNAVAILABLE,
                "none",
                reason=f"{reason}; LLM also failed: {error}",
                elapsed=time.perf_counter() - started,
            )
        return Answer(
            query, text, ROUTE_FALLBACK, "LLM", reason=reason, elapsed=time.perf_counter() - started
        )

    # ------------------------------------------------------------------ logging

    def _record(self, answer):
        self.log.append(answer)
        if self.verbose:
            print(format_record(answer))
        return answer

    def route_counts(self):
        counts = {}
        for answer in self.log:
            counts[answer.route] = counts.get(answer.route, 0) + 1
        return counts


def format_record(answer):
    """One log line per answered question, route first."""
    return (
        f"[{answer.route:<11} via {answer.source:<10}] "
        f"{answer.elapsed * 1000:7.1f} ms  {answer.reason}"
    )
