# Results — captured runs

Everything on this page came out of a real run on this machine. The raw terminal log is
[`docs/output.txt`](docs/output.txt).

**Environment**

| | |
|---|---|
| Model | `llama3.2:3b` via local Ollama (`ollama serve`) |
| Temperature | `0.1` |
| Dependencies | none — standard library only (`urllib`, `ast`, `math`, `concurrent.futures`) |
| Tests | 43, no model or network needed |

Reproduce with:

```bash
ollama serve && ollama pull llama3.2:3b
python3 run_demo.py
python3 tests.py
```

---

## Route table — 11 queries, all four routes exercised

| # | Route | Source | ms | Question |
|---|---|---|---|---|
| 1 | TOOL | Calculator | 1 | What is 12 * 8? |
| 2 | TOOL | Calculator | 0 | What is sqrt(144) + 5? |
| 3 | TOOL | Search | 0 | Tell me something about Tunisia |
| 4 | FALLBACK | LLM | 6838 | Search 404 test |
| 5 | FALLBACK | LLM | 8034 | Search timeout please |
| 6 | FALLBACK | LLM | 4287 | What is 5 / 0? |
| 7 | FALLBACK | LLM | 6884 | Search the mating habits of the Norwegian blue parrot |
| 8 | FALLBACK | LLM | 7747 | Tell me about quantum computing |
| 9 | DIRECT | LLM | 1438 | What is sqr(144)? |
| 10 | DIRECT | LLM | 5115 | Who are you? |
| 11 | DIRECT | LLM | 6978 | Why is fallback logic important for an agent? |

**TOOL 3, FALLBACK 5, DIRECT 3.** A tool answer costs about 1 ms; every route that reaches the
model costs 1.4–8 s. That gap is the argument for routing at all — the model is the expensive,
approximate path, and a calculator is exact and free.

`FALLBACK` and `DIRECT` both end at the model but are logged apart on purpose. `DIRECT` means no
tool was appropriate; `FALLBACK` means a tool was tried and did not deliver. Collapsing them
would hide tool breakage — five silent fallbacks in a row is an outage, five direct answers is a
normal day.

---

## The five fallbacks, and what triggered each

**1. Tool error — the backend returns 404**

```
Q: Search 404 test
ROUTE: FALLBACK via LLM
WHY  : Search failed: search backend returned HTTP 404 (index unavailable)
A: I don't have a specific "Search 404" test, but I can try to provide a general answer. [...]
```

**2. Tool timeout — the tool hangs past its deadline**

```
Q: Search timeout please
ROUTE: FALLBACK via LLM
WHY  : Search timed out: Search exceeded its 2.0s deadline
```

The deadline is enforced by the agent, not trusted to the tool. Measured: the agent returns at
**2001 ms** while the tool goes on sleeping to 2200 ms.

**3. Tool cannot answer — division by zero**

```
Q: What is 5 / 0?
ROUTE: FALLBACK via LLM
WHY  : Calculator failed: division by zero
A: [...] division by zero is undefined in mathematics. It's not a valid mathematical operation [...]
```

**4. Tool ran fine and found nothing — irrelevant query**

```
Q: Search the mating habits of the Norwegian blue parrot
ROUTE: FALLBACK via LLM
WHY  : Search failed: no results in the local index for that query
```

**5. Tool answered, but the answer was too thin to serve — the confidence fallback**

```
Q: Tell me about quantum computing
ROUTE: FALLBACK via LLM
WHY  : Search returned a weak answer (answer too short)
A: Quantum computing is a new approach to computing that uses the principles of quantum mechanics
   to perform calculations [...] Quantum computers use qubits, which can exist in multiple states
   simultaneously [...]
```

The index holds exactly `"It is a computing paradigm."` for that topic. Nothing failed, nothing
errored — the tool was just useless, which is the failure mode an error-only fallback misses.

---

## Three problems found while building this

### 1. `with ThreadPoolExecutor(...)` made the timeout cosmetic

The first version enforced the deadline inside a context manager. `future.result(timeout=2.0)`
raised on time, the fallback ran, the log said `timed out` — and the agent still took **2200 ms**,
the tool's full sleep. Leaving the `with` block calls `shutdown(wait=True)`, which joins the hung
worker thread. So the timeout was reported correctly and did nothing.

```
before (context manager):  2200.5 ms
after  (explicit shutdown(wait=False)):  2000.8 ms
```

A timeout you cannot measure is not a timeout. This one only showed up because the log printed
elapsed milliseconds next to the route.

### 2. An unknown function silently became its own argument

The expression extractor drops words it does not recognise, which is what lets it read a whole
question. It also meant a typo evaluated cleanly:

```
sqr(144)   ->  'sqr' dropped  ->  '(144)'  ->  144
```

`144` for a square root of 144. No error, no warning, and the wrong answer is the kind that looks
right — the real answer is 12. The fix is `unknown_function()`, which scans the query for
`name(` and refuses anything not implemented. `sqr(144)` now scores 0.0, so the router sends it to
the model, which answers *"The square root of 144 is 12."*

### 3. Keyword routing over-claimed on a reasoning question

`Why is fallback logic important for an agent?` contains the indexed word **agent**, so Search
scored 0.9 and won. It returned a correct passage that never answered the question:

```
before:  ROUTE: TOOL via Search   (1 ms)
         WHY  : matched topic 'agent'
         A: An AI agent perceives its environment, decides what to do, and acts towards a goal
            rather than answering one isolated question. [...]
```

Matching a keyword is not the same as answering the question. Search now caps its score at 0.3 for
questions starting *why / how / explain / should / compare*, unless the user explicitly asked to
search:

```
after:   ROUTE: DIRECT via LLM   (6978 ms)
         WHY  : no tool scored above 0.5 (best 0.30)
         A: Fallback logic is important for an agent because it allows the agent to recover from
            unexpected or unforeseen situations, ensuring it can continue to function and achieve
            its goals. [...]
```

Worth noting the cost direction: the fix made the answer 7000× slower and much better. Routing is
a quality decision before it is a speed one.

---

## Skill check

### Why is fallback logic essential for real-world agents?

Because a tool is an external dependency, and every external dependency fails eventually. Of the
five failure modes exercised above, only one is an exception — the other four are a hang, an empty
result set, a bad route, and a technically-successful-but-useless answer. An agent whose only plan
is "call the tool" turns each of those into either a crash or a confident piece of nonsense, and
the second is worse, because nothing in the output says anything went wrong.

Fallback also changes what the failure costs. With it, a broken search index downgrades the answer
from *sourced* to *from the model's own knowledge* — worse, but still an answer — and the log says
which one the user got. Without it, the same outage takes the whole feature down. The logging is
half the point: routes are the signal that tells you a tool is down at all, since a fallback
answer looks perfectly normal from the outside.

### Confidence-based fallback

Implemented, and it is the fifth trigger in the list above: a prose tool answer under 80
characters or under 0.5 confidence is treated as not worth serving.

The catch is that a naive length rule breaks the calculator. `96` is 2 characters and the complete,
exact answer — falling back there would replace an exact result with a guess. So `ToolResult`
carries an `atomic` flag for results that are short *by design*, and the length rule skips them.
There's a test for it (`test_short_calculator_answers_do_not_trigger_the_confidence_rule`),
because the failure would be invisible: the agent would still return the right number, just slower
and via the model.

---

## Tests

`python3 tests.py` → **43 tests, all passing**, with no model and no network. The LLM is replaced
by a stub that records the prompts it receives, which is what makes the fallback paths checkable —
including that the fallback prompt tells the model the tool was unavailable, and that a weak tool
fragment is passed as *possibly incomplete* rather than as fact.

Covered beyond the happy path: `eval` injection (`__import__('os').system(...)` is refused by the
`ast` walker), a tool raising an unexpected `KeyError`, the model being offline during a fallback
(route `UNAVAILABLE`, honest message, no invented answer), and the agent not waiting for a hung
tool.
