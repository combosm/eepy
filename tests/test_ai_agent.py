from __future__ import annotations

import os
import unittest
from unittest.mock import Mock, patch

try:
    from ai import ai_agent
except ImportError:
    ai_agent = None


@unittest.skipIf(ai_agent is None, "LangChain 1.x is not installed")
class AiAgentTests(unittest.TestCase):
    def test_invoke_agent_uses_messages_and_validates_structured_response(self) -> None:
        agent = Mock()
        agent.invoke.return_value = {
            "structured_response": {
                "topic": "driver safety",
                "summary": "Take a safe break.",
            }
        }
        history: list[dict[str, str]] = []

        response = ai_agent.invoke_agent(agent, "What should I do?", history)

        self.assertEqual(response.summary, "Take a safe break.")
        agent.invoke.assert_called_once_with(
            {
                "messages": [
                    {"role": "user", "content": "What should I do?"},
                ]
            }
        )
        self.assertEqual(history[-1]["role"], "assistant")

    def test_initialise_agent_uses_langchain_v1_structured_output(self) -> None:
        fake_model = Mock()
        fake_agent = Mock()

        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}),
            patch.object(ai_agent, "ChatOpenAI", return_value=fake_model),
            patch.object(ai_agent, "create_agent", return_value=fake_agent) as create,
        ):
            result = ai_agent.initialise_agent()

        self.assertIs(result, fake_agent)
        create.assert_called_once_with(
            model=fake_model,
            tools=[ai_agent.search_web, ai_agent.search_wikipedia],
            system_prompt=ai_agent.SYSTEM_PROMPT,
            response_format=ai_agent.ResearchResponse,
        )

    def test_initialise_agent_requires_openai_key(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "OPENAI_API_KEY"):
                ai_agent.initialise_agent()


if __name__ == "__main__":
    unittest.main()
