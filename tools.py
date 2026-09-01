"""The two tools the agent can reach for, and the errors they raise.

A tool is anything with a name, a description, a `score(query)` that says how
well it fits the question, and a `run(query)` that either returns a ToolResult
or raises a ToolError. The agent knows nothing else about them, so adding a
third tool means appending to TOOLS at the bottom of this file.
"""

import ast
import math
import operator
import re
import time


class ToolError(RuntimeError):
    """The tool ran and could not produce an answer."""


class ToolTimeout(ToolError):
    """The tool took longer than the agent was willing to wait."""


class ToolResult:
    """What a tool hands back.

    `confidence` is the tool's own opinion of its answer, 0.0 to 1.0.
    `atomic` marks a result that is short *by design* -- a calculator returning
    "96" is not a thin answer, it is the whole answer. Without that flag the
    "too short, fall back" rule would reject every correct sum.
    """

    def __init__(self, text, confidence=1.0, atomic=False, detail=""):
        self.text = text
        self.confidence = confidence
        self.atomic = atomic
        self.detail = detail

    def __repr__(self):
        return f"ToolResult({self.text!r}, confidence={self.confidence})"


# --------------------------------------------------------------------------
# Calculator
# --------------------------------------------------------------------------

_BINARY_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}

_FUNCTIONS = {
    "sqrt": math.sqrt,
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "pow": pow,
    "log": math.log,
    "log10": math.log10,
    "log2": math.log2,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "floor": math.floor,
    "ceil": math.ceil,
    "factorial": math.factorial,
}
_CONSTANTS = {"pi": math.pi, "e": math.e, "tau": math.tau}

# Words people write instead of symbols.
_WORD_OPERATORS = [
    (r"\bplus\b", "+"),
    (r"\bminus\b", "-"),
    (r"\btimes\b", "*"),
    (r"\bmultiplied by\b", "*"),
    (r"\bdivided by\b", "/"),
    (r"\bto the power of\b", "**"),
    (r"\bsquare root of\b", "sqrt"),
    (r"\bpercent of\b", "/100*"),
    (r"\^", "**"),
    (r"×", "*"),
    (r"÷", "/"),
]

_EXPRESSION_CHARS = re.compile(r"[0-9\.\+\-\*/%\(\)\s,]")


def extract_expression(query):
    """Pull the arithmetic out of a sentence.

    "What is sqrt(144) + 5?" -> "sqrt(144) + 5"
    Anything that is not a number, an operator, or a known function name is
    dropped, which is what lets the tool accept a question rather than only a
    bare expression.
    """
    text = query.lower().strip().rstrip("?.!")
    for pattern, replacement in _WORD_OPERATORS:
        text = re.sub(pattern, replacement, text)

    known = set(_FUNCTIONS) | set(_CONSTANTS)
    kept = []
    index = 0
    while index < len(text):
        matched_word = None
        for word in known:
            if not text.startswith(word, index):
                continue
            # Only at a word boundary, or "are you" would match the constant
            # `e` inside "ar|e" and leave a stray number-less fragment behind.
            before = text[index - 1] if index else " "
            after = text[index + len(word)] if index + len(word) < len(text) else " "
            if before.isalnum() or after.isalpha():
                continue
            matched_word = word
            break
        if matched_word:
            kept.append(matched_word)
            index += len(matched_word)
            continue

        character = text[index]
        if _EXPRESSION_CHARS.match(character):
            kept.append(character)
        elif character.isalpha():
            # An unknown word ends the expression fragment, so
            # "5 apples and 3 oranges" does not silently become "5 3".
            kept.append(" ")
        index += 1

    expression = "".join(kept)
    # Collapse the gaps left by dropped words and keep the longest fragment,
    # which is the one carrying the actual sum.
    fragments = [part.strip() for part in re.split(r"\s{2,}", expression) if part.strip()]
    if not fragments:
        return ""
    fragments.sort(key=lambda part: (sum(character.isdigit() for character in part), len(part)))
    return re.sub(r"\s+", " ", fragments[-1]).strip(" ,")


def _evaluate(node):
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ToolError(f"{node.value!r} is not a number")
    if isinstance(node, ast.BinOp):
        handler = _BINARY_OPS.get(type(node.op))
        if handler is None:
            raise ToolError("that operator is not supported")
        return handler(_evaluate(node.left), _evaluate(node.right))
    if isinstance(node, ast.UnaryOp):
        handler = _UNARY_OPS.get(type(node.op))
        if handler is None:
            raise ToolError("that operator is not supported")
        return handler(_evaluate(node.operand))
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCTIONS:
            raise ToolError("unknown function")
        return _FUNCTIONS[node.func.id](*[_evaluate(argument) for argument in node.args])
    if isinstance(node, ast.Name):
        if node.id in _CONSTANTS:
            return _CONSTANTS[node.id]
        raise ToolError(f"unknown name {node.id!r}")
    raise ToolError("that expression is not allowed")


def _format_number(value):
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise ToolError("the result is not a finite number")
        if value.is_integer():
            return str(int(value))
        return f"{round(value, 10):g}"
    return str(value)


_CALL_PATTERN = re.compile(r"\b([a-z_][a-z0-9_]*)\s*\(")


def unknown_function(query):
    """Return the first function name in the query that we do not implement.

    Needed because the extractor drops unknown words: `sqr(144)` would lose the
    `sqr` and evaluate to `144`, a wrong answer that looks like a right one.
    A typo has to be an error, not a silent result.
    """
    lowered = query.lower()
    for name in _CALL_PATTERN.findall(lowered):
        if name not in _FUNCTIONS:
            return name
    return None


class Calculator:
    """Evaluates arithmetic with `ast`, never `eval`."""

    name = "Calculator"
    description = "Evaluates arithmetic: + - * / ** %, sqrt, log, round, min, max, factorial."

    def score(self, query):
        if unknown_function(query):
            # Refuse the question rather than answering a mangled version of it.
            return 0.0
        expression = extract_expression(query)
        if not expression or not any(character.isdigit() for character in expression):
            return 0.0
        has_operator = any(symbol in expression for symbol in "+-*/%")
        has_function = any(name + "(" in expression for name in _FUNCTIONS)
        if not (has_operator or has_function):
            return 0.0
        # A whole sentence with one number and one dash is probably prose.
        return 0.95 if (has_function or len(re.findall(r"\d", expression)) >= 2) else 0.6

    def run(self, query):
        missing = unknown_function(query)
        if missing:
            raise ToolError(f"unknown function {missing!r}")
        expression = extract_expression(query)
        if not expression:
            raise ToolError("no arithmetic found in the question")
        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError as error:
            raise ToolError(f"{expression!r} is not a valid expression") from error

        try:
            value = _evaluate(tree)
        except ZeroDivisionError as error:
            raise ToolError("division by zero") from error
        except ValueError as error:
            # sqrt(-4), log(0), factorial(-1) all land here.
            raise ToolError(f"undefined for those values ({error})") from error
        except OverflowError as error:
            raise ToolError("the result is too large to represent") from error

        return ToolResult(
            _format_number(value),
            confidence=1.0,
            atomic=True,
            detail=f"evaluated {expression}",
        )


# --------------------------------------------------------------------------
# Search
# --------------------------------------------------------------------------

# A tiny offline corpus. Real search would be an HTTP call; the failure modes
# are what this checkpoint is about, and those are easier to trigger on demand.
CORPUS = {
    "tunisia": (
        "Tunisia is a North African country on the Mediterranean coast, bordered by Algeria to "
        "the west and Libya to the south-east. Its capital is Tunis. Carthage, just outside the "
        "capital, was the centre of the Carthaginian empire and is now a UNESCO World Heritage "
        "site. Arabic is the official language and French is widely used in business."
    ),
    "morocco": (
        "Morocco is a North African country with coastlines on both the Atlantic Ocean and the "
        "Mediterranean Sea. Rabat is the capital and Casablanca the largest city. The Atlas "
        "Mountains run through the country and separate the coast from the Sahara."
    ),
    "ollama": (
        "Ollama runs open-weight language models locally. It exposes an HTTP API on port 11434 "
        "with endpoints such as /api/generate, /api/chat and /api/embed, so an application talks "
        "to a local model the same way it would talk to a hosted one -- without an API key."
    ),
    "langchain": (
        "LangChain organises the building blocks of an LLM application: prompts, models, tools, "
        "memory and chains. Its LCEL syntax composes those blocks with the pipe operator, so a "
        "prompt piped into a model piped into a parser is itself one runnable object."
    ),
    "agent": (
        "An AI agent perceives its environment, decides what to do, and acts towards a goal "
        "rather than answering one isolated question. Tool use is what separates an agent from a "
        "chatbot: it can call a calculator, a search index or an API instead of guessing."
    ),
    # Deliberately thin: one clause, no substance. Used to trigger the
    # confidence fallback rather than an error.
    "quantum computing": "It is a computing paradigm.",
}

_STOP_WORDS = {
    "a", "about", "an", "and", "any", "are", "can", "do", "does", "for", "find",
    "from", "give", "how", "i", "in", "is", "it", "know", "look", "lookup", "me",
    "more", "of", "on", "please", "search", "something", "tell", "the", "up",
    "what", "who", "why", "you", "your",
}


# Questions that ask for reasoning rather than a lookup. A keyword match is not
# enough for these: "Why is fallback logic important for an agent?" contains the
# word "agent" and matched the index, which returned a passage defining agents
# and never answered the "why".
_REASONING_STARTS = (
    "why", "how", "explain", "should", "compare", "what do you think",
    "what happens if", "is it better",
)


def _keywords(query):
    words = re.findall(r"[a-z0-9']+", query.lower())
    return [word for word in words if word not in _STOP_WORDS and len(word) > 1]


class Search:
    """Looks a topic up in a small local index.

    Two failures can be triggered on purpose, because the brief asks for them:
    a query containing "404" returns a backend error, and one containing
    "timeout" hangs past the agent's deadline.
    """

    name = "Search"
    description = "Looks up factual topics in a local knowledge index."

    # How long the agent is willing to wait for this tool.
    timeout_seconds = 2.0

    def __init__(self, corpus=None):
        self.corpus = corpus if corpus is not None else CORPUS

    def score(self, query):
        lowered = query.lower()
        keywords = _keywords(query)
        asked_explicitly = lowered.startswith(("search", "look up", "find")) or "search" in lowered
        wants_a_topic = any(
            phrase in lowered for phrase in ("tell me about", "tell me something about", "what is", "who is")
        )

        reasoning = lowered.startswith(_REASONING_STARTS)

        if self._best_topic(keywords):
            # A reasoning question that happens to name an indexed topic still
            # is not a lookup, so keep it under the threshold and let the model
            # answer it.
            return 0.3 if (reasoning and not asked_explicitly) else 0.9
        if asked_explicitly:
            # Honour the explicit request even with nothing to match on -- that
            # is what produces the "no results" fallback instead of a silent
            # refusal to route.
            return 0.7
        if wants_a_topic and keywords:
            return 0.4
        return 0.0

    def _best_topic(self, keywords):
        for topic in self.corpus:
            topic_words = set(topic.split())
            if topic_words & set(keywords):
                return topic
        return None

    def run(self, query):
        lowered = query.lower()

        if "404" in lowered:
            raise ToolError("search backend returned HTTP 404 (index unavailable)")

        if "timeout" in lowered:
            # Sleep past the deadline the agent gave us, then say so. A real
            # client would be cut off by its own socket timeout.
            time.sleep(self.timeout_seconds + 0.2)
            raise ToolTimeout(f"search did not respond within {self.timeout_seconds}s")

        topic = self._best_topic(_keywords(query))
        if topic is None:
            raise ToolError("no results in the local index for that query")

        passage = self.corpus[topic]
        # Confidence drops when the stored passage is barely anything. This is
        # what the confidence fallback keys on.
        words = len(passage.split())
        confidence = 1.0 if words >= 40 else (0.6 if words >= 15 else 0.2)
        return ToolResult(passage, confidence=confidence, detail=f"matched topic {topic!r}")


TOOLS = [Calculator(), Search()]
