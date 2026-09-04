from __future__ import annotations

import unittest
from types import SimpleNamespace

from web.algorithm_registry import (
    AgentResponseError,
    describe_agent_error,
    get_agent_configuration,
    suggest_algorithm_spec,
)
from web.algorithm_registry.agent_spec import (
    AlgorithmSpecAgentOutput,
    EquationSuggestion,
    EvidenceSuggestion,
    HyperparameterSuggestion,
    MAX_AGENT_SOURCE_CHARS,
    PseudocodeStepSuggestion,
)


class FakeStructuredModel:
    def __init__(self, response):
        self.response = response
        self.messages = None

    def invoke(self, messages):
        self.messages = messages
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


class FakeChatModel:
    def __init__(self, response):
        self.structured_request = None
        self.structured_model = FakeStructuredModel(response)

    def with_structured_output(self, schema, **kwargs):
        self.structured_request = {"schema": schema, **kwargs}
        return self.structured_model


class FallbackChatModel:
    def __init__(self, successful_response):
        self.successful_response = successful_response
        self.methods = []
        self.structured_models = []

    def with_structured_output(self, schema, **kwargs):
        method = kwargs["method"]
        self.methods.append(method)
        if method == "function_calling":
            structured_model = FakeStructuredModel(
                RuntimeError(
                    "tool_choice parameter does not support required in "
                    "thinking mode"
                )
            )
        else:
            structured_model = FakeStructuredModel(self.successful_response)
        self.structured_models.append(structured_model)
        return structured_model


def agent_output(source_excerpt: str) -> AlgorithmSpecAgentOutput:
    return AlgorithmSpecAgentOutput(
        algorithm_id="Q Learning",
        name="Q-Learning",
        category="value-based control",
        summary="Learn action values with temporal-difference targets.",
        objective="Learn a greedy policy from one-step transitions.",
        assumptions=["Finite action space"],
        inputs=["Environment transitions"],
        outputs=["Action-value function", "Policy"],
        states=["Environment state"],
        actions=["Available environment action"],
        hyperparameters=[
            HyperparameterSuggestion(
                name="gamma",
                value_type="number",
                default_value="0.99",
                description="Discount factor.",
                minimum=0.0,
                maximum=1.0,
                step=0.01,
            ),
            HyperparameterSuggestion(
                name="episodes",
                value_type="integer",
                default_value="1000",
                description="Training episodes.",
            ),
        ],
        core_equations=[
            EquationSuggestion(
                latex=r"Q(s,a) \leftarrow Q(s,a) + \alpha(r + \gamma \max Q - Q(s,a))",
                explanation="One-step temporal-difference update.",
            )
        ],
        pseudocode=[
            PseudocodeStepSuggestion(instruction="Observe a transition"),
            PseudocodeStepSuggestion(instruction="Update the action value"),
        ],
        supported_environments=["FrozenLake-v1"],
        evidence=[
            EvidenceSuggestion(
                supports_fields=["objective", "core_equations"],
                source_excerpt=source_excerpt,
                explanation="The source states the one-step update.",
            ),
            EvidenceSuggestion(
                supports_fields=["summary"],
                source_excerpt="This quotation is not in the source.",
                explanation="Unverifiable evidence.",
            ),
        ],
        warnings=["Confirm the environment name before publishing."],
    )


class AlgorithmSpecAgentTests(unittest.TestCase):
    def test_provider_500_is_presented_as_manual_retry_error(self) -> None:
        error = AgentResponseError(
            "Error code: 500 - {'error': {'code': 'internal_server_error'}, "
            "'request_id': '061a929f-9983-9057-a230-61d3c51c6973'}"
        )
        details = describe_agent_error(error)
        self.assertIn("No draft file was changed", details["summary"])
        self.assertIn("provider-side failure", details["summary"])
        self.assertIn("Retry generation", details["summary"])
        self.assertEqual(
            details["request_id"], "061a929f-9983-9057-a230-61d3c51c6973"
        )
        self.assertIn("Error code: 500", details["technical"])

    def test_configuration_requires_only_environment_key(self) -> None:
        missing = get_agent_configuration({})
        configured = get_agent_configuration(
            {
                "ALGORITHM_AGENT_API_KEY": "test-key",
                "ALGORITHM_AGENT_BASE_URL": "https://example.test/v1",
                "ALGORITHM_AGENT_MODEL": "replaceable-model",
            }
        )

        self.assertFalse(missing.configured)
        self.assertTrue(configured.configured)
        self.assertEqual(configured.model, "replaceable-model")
        self.assertEqual(configured.base_url, "https://example.test/v1")
        self.assertEqual(
            configured.structured_output_method,
            "function_calling",
        )

    def test_configuration_supports_legacy_names_and_rejects_bad_method(
        self,
    ) -> None:
        legacy = get_agent_configuration(
            {
                "OPENAI_API_KEY": "test-key",
                "OPENAI_BASE_URL": "https://legacy.example/v1",
                "OPENAI_ALGORITHM_SPEC_MODEL": "legacy-model",
                "ALGORITHM_AGENT_STRUCTURED_METHOD": "json_schema",
            }
        )
        invalid = get_agent_configuration(
            {
                "ALGORITHM_AGENT_API_KEY": "test-key",
                "ALGORITHM_AGENT_STRUCTURED_METHOD": "xml",
            }
        )

        self.assertTrue(legacy.configured)
        self.assertEqual(legacy.model, "legacy-model")
        self.assertEqual(legacy.base_url, "https://legacy.example/v1")
        self.assertEqual(legacy.structured_output_method, "json_schema")
        self.assertFalse(invalid.configured)
        self.assertIn("must be one of", invalid.message)

    def test_structured_agent_output_is_normalized_and_evidence_verified(self) -> None:
        excerpt = "Q-learning updates an action value from a one-step target."
        source = f"# Q-Learning\n\n{excerpt}\n"
        response = {
            "raw": SimpleNamespace(
                id="run_test",
                usage_metadata={
                    "input_tokens": 120,
                    "output_tokens": 80,
                    "total_tokens": 200,
                },
            ),
            "parsed": agent_output(excerpt),
            "parsing_error": None,
        }
        chat_model = FakeChatModel(response)

        suggestion = suggest_algorithm_spec(
            "q-learning.md",
            source,
            chat_model=chat_model,
            model="test-model",
        )

        self.assertEqual(suggestion.values["algorithm_id"], "q-learning")
        self.assertEqual(
            suggestion.values["hyperparameters"]["gamma"]["default"],
            0.99,
        )
        self.assertEqual(
            suggestion.values["hyperparameters"]["episodes"]["default"],
            1000,
        )
        self.assertEqual(
            suggestion.values["hyperparameters"]["gamma"]["maximum"], 1.0
        )
        self.assertEqual(len(suggestion.evidence), 1)
        self.assertEqual(suggestion.evidence[0]["verification"], "exact")
        self.assertTrue(
            any("could not be verified" in item for item in suggestion.warnings)
        )
        self.assertEqual(suggestion.usage["total_tokens"], 200)
        self.assertEqual(
            suggestion.provider,
            "langchain-openai-compatible",
        )
        self.assertEqual(
            suggestion.structured_output_method,
            "function_calling",
        )
        self.assertIs(
            chat_model.structured_request["schema"],
            AlgorithmSpecAgentOutput,
        )
        self.assertEqual(
            chat_model.structured_request["method"],
            "function_calling",
        )
        self.assertTrue(chat_model.structured_request["include_raw"])
        self.assertEqual(
            chat_model.structured_model.messages[0][0],
            "system",
        )
        self.assertIn(
            "Regardless of the language used by the source",
            chat_model.structured_model.messages[0][1],
        )
        self.assertIn(
            "Evidence is the only language exception",
            chat_model.structured_model.messages[0][1],
        )

    def test_line_wrapped_source_evidence_is_verified_without_paraphrasing(
        self,
    ) -> None:
        model_excerpt = (
            "Q-learning updates an action value from a one-step target."
        )
        source = (
            "# Q-Learning\n\nQ-learning updates an action value from a\n"
            "one-step target.\n"
        )
        chat_model = FakeChatModel(
            {
                "raw": SimpleNamespace(
                    id="run_wrapped", usage_metadata=None
                ),
                "parsed": agent_output(model_excerpt),
                "parsing_error": None,
            }
        )

        suggestion = suggest_algorithm_spec(
            "wrapped.md",
            source,
            chat_model=chat_model,
        )

        self.assertEqual(len(suggestion.evidence), 1)
        self.assertEqual(
            suggestion.evidence[0]["verification"],
            "whitespace_normalized",
        )
        self.assertIn("\n", suggestion.evidence[0]["source_excerpt"])

    def test_long_sources_are_bounded_and_reported(self) -> None:
        source = "A" * (MAX_AGENT_SOURCE_CHARS + 1_000)
        response = {
            "raw": SimpleNamespace(id="run_long", usage_metadata=None),
            "parsed": agent_output("A" * 40),
            "parsing_error": None,
        }
        chat_model = FakeChatModel(response)

        suggestion = suggest_algorithm_spec(
            "long.txt",
            source,
            chat_model=chat_model,
        )

        self.assertTrue(suggestion.source_truncated)
        self.assertLessEqual(
            suggestion.submitted_characters,
            MAX_AGENT_SOURCE_CHARS + 100,
        )
        self.assertTrue(
            any("60,000 characters" in item for item in suggestion.warnings)
        )

    def test_tool_choice_incompatibility_requires_manual_retry(self) -> None:
        excerpt = "Q-learning updates an action value from a one-step target."
        response = {
            "raw": SimpleNamespace(id="run_fallback", usage_metadata=None),
            "parsed": agent_output(excerpt),
            "parsing_error": None,
        }
        chat_model = FallbackChatModel(response)

        with self.assertRaises(AgentResponseError):
            suggest_algorithm_spec(
                "q-learning.md",
                excerpt,
                chat_model=chat_model,
            )

        self.assertEqual(
            chat_model.methods,
            ["function_calling"],
        )

    def test_chinese_evidence_is_preserved_and_translation_is_rejected(self) -> None:
        source = "普通 Q-Learning 容易出现系统性的高估。"
        output = agent_output(source)
        output.evidence.append(
            EvidenceSuggestion(
                supports_fields=["summary"],
                source_excerpt="Standard Q-Learning can systematically overestimate values.",
                explanation="Translated evidence must not pass verification.",
            )
        )
        chat_model = FakeChatModel(
            {"raw": SimpleNamespace(id="run_cn", usage_metadata=None), "parsed": output}
        )

        suggestion = suggest_algorithm_spec("source.md", source, chat_model=chat_model)

        self.assertEqual(len(suggestion.evidence), 1)
        self.assertEqual(suggestion.evidence[0]["source_excerpt"], source)
        self.assertTrue(suggestion.platform_warnings)

    def test_missing_structured_output_is_rejected(self) -> None:
        chat_model = FakeChatModel(
            {
                "raw": SimpleNamespace(id="run_empty", usage_metadata=None),
                "parsed": None,
                "parsing_error": None,
            }
        )

        with self.assertRaises(AgentResponseError):
            suggest_algorithm_spec(
                "source.md",
                "A valid source.",
                chat_model=chat_model,
            )


if __name__ == "__main__":
    unittest.main()
