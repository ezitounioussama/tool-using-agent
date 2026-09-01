"""Tests for the agent, the router and both tools. No model, no network.

The LLM is replaced by a stub that records the prompts it was given, which is
how the fallback paths can be checked without Ollama running.
"""

import time
import unittest

import agent as agent_module
from agent import (
    Agent,
    ROUTE_DIRECT,
    ROUTE_FALLBACK,
    ROUTE_TOOL,
    ROUTE_UNAVAILABLE,
)
from llm import LLMError
from tools import (
    Calculator,
    Search,
    ToolError,
    ToolResult,
    ToolTimeout,
    extract_expression,
)


class StubModel:
    """Stands in for the LLM. Remembers every prompt it received."""

    def __init__(self, reply="[model answer]", fail=False):
        self.reply = reply
        self.fail = fail
        self.prompts = []

    def __call__(self, prompt, system=None):
        self.prompts.append(prompt)
        if self.fail:
            raise LLMError("stub model is offline")
        return self.reply


def build_agent(**kwargs):
    model = StubModel(**kwargs)
    return Agent(generate=model, verbose=False), model


# ---------------------------------------------------------------- Calculator


class TestExpressionExtraction(unittest.TestCase):
    def test_pulls_arithmetic_out_of_a_question(self):
        self.assertEqual(extract_expression("What is 12 * 8?"), "12 * 8")

    def test_keeps_function_calls(self):
        self.assertEqual(extract_expression("What is sqrt(144) + 5?"), "sqrt(144) + 5")

    def test_translates_written_operators(self):
        self.assertEqual(extract_expression("what is 2 to the power of 10"), "2 ** 10")
        self.assertEqual(extract_expression("7 times 6"), "7 * 6")

    def test_no_arithmetic_gives_empty_string(self):
        self.assertEqual(extract_expression("Who are you?"), "")
        self.assertEqual(extract_expression("Tell me something about Tunisia"), "")

    def test_constant_only_matches_on_a_word_boundary(self):
        # 'e' is a known constant; it must not be picked out of "are".
        self.assertNotIn("e", extract_expression("Who are you?"))
        self.assertEqual(extract_expression("what is e * 2"), "e * 2")

    def test_prose_with_numbers_is_not_an_expression(self):
        self.assertEqual(Calculator().score("5 apples and 3 oranges"), 0.0)


class TestCalculator(unittest.TestCase):
    def setUp(self):
        self.calculator = Calculator()

    def test_multiplication(self):
        self.assertEqual(self.calculator.run("What is 12 * 8?").text, "96")

    def test_sqrt_and_addition(self):
        self.assertEqual(self.calculator.run("What is sqrt(144) + 5?").text, "17")

    def test_whole_number_floats_lose_the_decimal(self):
        self.assertEqual(self.calculator.run("10 / 2").text, "5")

    def test_real_decimals_are_kept(self):
        self.assertEqual(self.calculator.run("7 / 2").text, "3.5")

    def test_division_by_zero_is_a_tool_error(self):
        with self.assertRaises(ToolError):
            self.calculator.run("What is 5 / 0?")

    def test_sqrt_of_a_negative_is_a_tool_error(self):
        with self.assertRaises(ToolError):
            self.calculator.run("sqrt(-4)")

    def test_unknown_function_is_refused(self):
        with self.assertRaises(ToolError):
            self.calculator.run("hack(2)")

    def test_a_typo_in_a_function_name_is_an_error_not_a_wrong_answer(self):
        # `sqr` is not `sqrt`. Before this check the name was dropped and the
        # expression evaluated to 144 -- a wrong answer that looks right.
        with self.assertRaises(ToolError):
            self.calculator.run("sqr(144)")
        self.assertEqual(self.calculator.score("sqr(144)"), 0.0)

    def test_no_arithmetic_is_a_tool_error(self):
        with self.assertRaises(ToolError):
            self.calculator.run("Who are you?")

    def test_result_is_marked_atomic(self):
        # Short by design, so the "too short" fallback must not fire on it.
        self.assertTrue(self.calculator.run("2 + 2").atomic)

    def test_never_evaluates_arbitrary_code(self):
        with self.assertRaises(ToolError):
            self.calculator.run("__import__('os').system('echo pwned')")


# -------------------------------------------------------------------- Search


class TestSearch(unittest.TestCase):
    def setUp(self):
        self.search = Search()

    def test_finds_a_topic_in_the_index(self):
        result = self.search.run("Tell me something about Tunisia")
        self.assertIn("Tunisia", result.text)
        self.assertEqual(result.confidence, 1.0)

    def test_404_query_raises_a_tool_error(self):
        with self.assertRaises(ToolError):
            self.search.run("Search 404 test")

    def test_timeout_query_raises_a_tool_timeout(self):
        self.search.timeout_seconds = 0.05
        with self.assertRaises(ToolTimeout):
            self.search.run("search timeout please")

    def test_unknown_topic_raises_a_tool_error(self):
        with self.assertRaises(ToolError):
            self.search.run("search the mating habits of the Norwegian blue parrot")

    def test_thin_passage_gets_low_confidence(self):
        result = self.search.run("Tell me about quantum computing")
        self.assertLess(result.confidence, agent_module.MIN_CONFIDENCE)

    def test_reasoning_question_does_not_route_to_search(self):
        # Contains the indexed word "agent", but asks why, not what.
        score = self.search.score("Why is fallback logic important for an agent?")
        self.assertLess(score, agent_module.SCORE_THRESHOLD)

    def test_lookup_question_on_the_same_topic_still_routes_to_search(self):
        self.assertGreaterEqual(
            self.search.score("What is an agent?"), agent_module.SCORE_THRESHOLD
        )

    def test_scores_an_explicit_search_request_even_with_no_match(self):
        self.assertGreaterEqual(self.search.score("Search 404 test"), agent_module.SCORE_THRESHOLD)


# --------------------------------------------------------------------- router


class TestRouting(unittest.TestCase):
    def setUp(self):
        self.agent, self.model = build_agent()

    def test_arithmetic_routes_to_the_calculator(self):
        tool, score = self.agent.choose_tool("What is 12 * 8?")
        self.assertEqual(tool.name, "Calculator")
        self.assertGreaterEqual(score, agent_module.SCORE_THRESHOLD)

    def test_topic_question_routes_to_search(self):
        tool, _ = self.agent.choose_tool("Tell me something about Tunisia")
        self.assertEqual(tool.name, "Search")

    def test_identity_question_routes_to_no_tool(self):
        tool, score = self.agent.choose_tool("Who are you?")
        self.assertIsNone(tool)
        self.assertLess(score, agent_module.SCORE_THRESHOLD)


# ---------------------------------------------------------------------- agent


class TestAgentRoutes(unittest.TestCase):
    def test_tool_route_returns_the_tool_answer(self):
        agent, model = build_agent()
        answer = agent.answer("What is 12 * 8?")
        self.assertEqual(answer.route, ROUTE_TOOL)
        self.assertEqual(answer.source, "Calculator")
        self.assertEqual(answer.text, "96")
        self.assertEqual(model.prompts, [], "the model must not be called when a tool succeeds")

    def test_tool_error_falls_back_to_the_model(self):
        agent, model = build_agent()
        answer = agent.answer("Search 404 test")
        self.assertEqual(answer.route, ROUTE_FALLBACK)
        self.assertEqual(answer.source, "LLM")
        self.assertIn("404", answer.reason)
        self.assertEqual(len(model.prompts), 1)

    def test_division_by_zero_falls_back(self):
        agent, _ = build_agent()
        answer = agent.answer("What is 5 / 0?")
        self.assertEqual(answer.route, ROUTE_FALLBACK)
        self.assertIn("division by zero", answer.reason)

    def test_timeout_falls_back_and_does_not_wait_for_the_tool(self):
        search = Search()
        search.timeout_seconds = 0.2
        agent = Agent(tools=[search], generate=StubModel(), verbose=False)

        started = time.perf_counter()
        answer = agent.answer("search timeout please")
        elapsed = time.perf_counter() - started

        self.assertEqual(answer.route, ROUTE_FALLBACK)
        self.assertIn("timed out", answer.reason)
        # The tool sleeps deadline + 0.2s. The agent must return on the deadline,
        # not after the tool finally gives up.
        self.assertLess(elapsed, 0.38, "the agent waited for the hung tool")

    def test_no_suitable_tool_answers_directly(self):
        agent, model = build_agent()
        answer = agent.answer("Who are you?")
        self.assertEqual(answer.route, ROUTE_DIRECT)
        self.assertEqual(len(model.prompts), 1)

    def test_direct_and_fallback_are_different_routes(self):
        agent, _ = build_agent()
        direct = agent.answer("Who are you?")
        fallback = agent.answer("Search 404 test")
        self.assertNotEqual(direct.route, fallback.route)

    def test_thin_tool_answer_falls_back_on_confidence(self):
        agent, model = build_agent()
        answer = agent.answer("Tell me about quantum computing")
        self.assertEqual(answer.route, ROUTE_FALLBACK)
        self.assertIn("weak answer", answer.reason)
        self.assertIn("quantum", model.prompts[0].lower())

    def test_short_calculator_answers_do_not_trigger_the_confidence_rule(self):
        agent, model = build_agent()
        answer = agent.answer("2 + 2")
        self.assertEqual(answer.route, ROUTE_TOOL)
        self.assertEqual(answer.text, "4")
        self.assertEqual(model.prompts, [])

    def test_a_tool_crashing_unexpectedly_still_falls_back(self):
        class Exploding:
            name = "Exploding"
            description = "always raises"

            def score(self, query):
                return 1.0

            def run(self, query):
                raise KeyError("boom")

        agent = Agent(tools=[Exploding()], generate=StubModel(), verbose=False)
        answer = agent.answer("anything")
        self.assertEqual(answer.route, ROUTE_FALLBACK)
        self.assertIn("KeyError", answer.reason)

    def test_model_offline_after_a_tool_failure_is_reported_honestly(self):
        agent, _ = build_agent(fail=True)
        answer = agent.answer("Search 404 test")
        self.assertEqual(answer.route, ROUTE_UNAVAILABLE)
        self.assertIn("also unavailable", answer.text)

    def test_model_offline_with_no_tool_is_reported_honestly(self):
        agent, _ = build_agent(fail=True)
        answer = agent.answer("Who are you?")
        self.assertEqual(answer.route, ROUTE_UNAVAILABLE)
        self.assertIn("unavailable", answer.text)

    def test_fallback_prompt_says_the_tool_was_unavailable(self):
        agent, model = build_agent()
        agent.answer("Search 404 test")
        self.assertIn("Search", model.prompts[0])
        self.assertIn("unavailable", model.prompts[0])

    def test_fallback_prompt_flags_a_weak_fragment_as_unreliable(self):
        agent, model = build_agent()
        agent.answer("Tell me about quantum computing")
        self.assertIn("may be incomplete", model.prompts[0])

    def test_every_answer_is_logged_with_its_route(self):
        agent, _ = build_agent()
        for query in ["What is 12 * 8?", "Search 404 test", "Who are you?"]:
            agent.answer(query)
        self.assertEqual(agent.route_counts(), {ROUTE_TOOL: 1, ROUTE_FALLBACK: 1, ROUTE_DIRECT: 1})


class TestToolResult(unittest.TestCase):
    def test_defaults(self):
        result = ToolResult("text")
        self.assertEqual(result.confidence, 1.0)
        self.assertFalse(result.atomic)


if __name__ == "__main__":
    unittest.main(verbosity=2)
