"""Runs the agent over the checkpoint's test queries and prints every route.

Each block prints the question, the route the agent took, and the answer, so the
tool / fallback / direct split is visible without reading any code.
"""

import sys

import llm
from agent import Agent, ROUTE_DIRECT, ROUTE_FALLBACK, ROUTE_TOOL, ROUTE_UNAVAILABLE

# (query, what it is meant to exercise)
QUERIES = [
    ("What is 12 * 8?", "tool: plain arithmetic"),
    ("What is sqrt(144) + 5?", "tool: function plus arithmetic"),
    ("Tell me something about Tunisia", "tool: topic found in the index"),
    ("Search 404 test", "fallback: the search backend errors"),
    ("Search timeout please", "fallback: the search tool hangs past its deadline"),
    ("What is 5 / 0?", "fallback: the calculator cannot answer"),
    ("Search the mating habits of the Norwegian blue parrot", "fallback: no results in the index"),
    ("Tell me about quantum computing", "fallback: tool answered, but too thin to serve"),
    ("What is sqr(144)?", "no tool: a typo must not become a wrong answer"),
    ("Who are you?", "direct: no tool applies"),
    ("Why is fallback logic important for an agent?", "direct: reasoning, not lookup"),
]

RULE = "=" * 78


def main():
    ok, message = llm.health()
    print(message)
    if not ok:
        return 1
    print(f"{RULE}\n TOOL-USING AGENT WITH FALLBACK LOGIC\n{RULE}")

    agent = Agent(verbose=False)

    for index, (query, purpose) in enumerate(QUERIES, start=1):
        print(f"\n--- {index}. {purpose}")
        print(f"    Q: {query}")
        answer = agent.answer(query)
        print(f"    ROUTE: {answer.route} via {answer.source}   ({answer.elapsed * 1000:.0f} ms)")
        print(f"    WHY  : {answer.reason}")
        text = " ".join(answer.text.split())
        print(f"    A: {text}")

    print(f"\n{RULE}\n SUMMARY\n{RULE}")
    counts = agent.route_counts()
    for route in (ROUTE_TOOL, ROUTE_FALLBACK, ROUTE_DIRECT, ROUTE_UNAVAILABLE):
        if counts.get(route):
            print(f"  {route:<12} {counts[route]:>2}")
    print(f"  {'TOTAL':<12} {len(agent.log):>2}")

    print("\n  route table")
    print(f"  {'#':<3} {'route':<12} {'source':<11} {'ms':>7}  question")
    for index, answer in enumerate(agent.log, start=1):
        question = answer.query if len(answer.query) <= 44 else answer.query[:41] + "..."
        print(
            f"  {index:<3} {answer.route:<12} {answer.source:<11} "
            f"{answer.elapsed * 1000:7.0f}  {question}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
