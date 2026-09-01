# Tool-Using Agent with Fallback Logic

An agent with two tools — a calculator and a small search index — that decides for each question
whether to reach for a tool or just answer, and never gets stuck when a tool lets it down.

The interesting half is what happens when the tool doesn't deliver, and "doesn't deliver" turns
out to have five shapes: it errors, it hangs past its deadline, it can't compute the answer, it
finds nothing, or it answers successfully with something too thin to be worth serving. That last
one is the reason a confidence check sits next to the error handling — nothing failed, so an
error-only fallback would have shipped `"It is a computing paradigm."` as an answer about quantum
computing.

Every answer is logged with the route that produced it, and `FALLBACK` is kept separate from
`DIRECT` on purpose: one means a tool broke, the other means no tool was needed. Five fallbacks in
a row is an outage; five direct answers is a normal day.

Runs locally on Ollama, no API key, standard library only.

```bash
ollama serve && ollama pull llama3.2:3b

python3 run_demo.py     # 11 queries, all four routes
python3 tests.py        # 43 tests, no model needed
```

## Also in this repo

- **[RESULTS.md](RESULTS.md)** — the captured run with every route, the five fallback triggers,
  three bugs found while building (including a timeout that reported correctly and did nothing),
  and the skill-check answers
- [`docs/output.txt`](docs/output.txt) — raw terminal log

---

Author: **Oussama Ezitouni**
