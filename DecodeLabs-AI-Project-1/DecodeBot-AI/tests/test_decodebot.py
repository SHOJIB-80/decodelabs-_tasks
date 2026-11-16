from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from conversation import ConversationSession, EMPTY_MESSAGE
from nlp_engine import IntentEngine, KnowledgeBase, KnowledgeBaseError, normalize_text


class TextNormalizationTests(unittest.TestCase):
    def test_lowercase_punctuation_and_whitespace(self):
        self.assertEqual(normalize_text("  WHAT'S   AI?!  "), "what is ai")

    def test_unicode_and_numbers_are_retained(self):
        self.assertEqual(normalize_text("Café, Python 3!"), "café python 3")

    def test_non_string_is_rejected(self):
        with self.assertRaises(TypeError):
            normalize_text(None)  # type: ignore[arg-type]


class KnowledgeBaseTests(unittest.TestCase):
    def test_valid_knowledge_base_loads_and_contains_multiple_examples(self):
        knowledge = KnowledgeBase.load(ROOT / "knowledge_base" / "intents.json")
        self.assertGreaterEqual(len(knowledge.intents), 10)
        self.assertTrue(all(len(item["examples"]) >= 2 for item in knowledge.intents.values()))

    def test_missing_file_has_actionable_error(self):
        with self.assertRaisesRegex(KnowledgeBaseError, "not found"):
            KnowledgeBase.load(ROOT / "missing-knowledge.json")

    def test_invalid_json_has_actionable_error(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bad.json"
            path.write_text("{bad", encoding="utf-8")
            with self.assertRaisesRegex(KnowledgeBaseError, "Invalid JSON"):
                KnowledgeBase.load(path)

    def test_invalid_intent_schema_is_rejected(self):
        with self.assertRaisesRegex(KnowledgeBaseError, "at least two"):
            KnowledgeBase({"intents": [{"name": "one", "examples": ["only one"], "responses": ["reply"]}]})


class IntentMatchingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = IntentEngine(KnowledgeBase.load(ROOT / "knowledge_base" / "intents.json"))

    def test_paraphrases_map_to_expected_intents(self):
        cases = {
            "could you describe AI in simple terms": "artificial_intelligence",
            "what kinds of tasks can Python handle": "python",
            "can you assist me with a bug in my script": "project_assistance",
            "how do computers learn from examples": "machine_learning",
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                match = self.engine.match(text)
                self.assertIsNotNone(match)
                self.assertEqual(match.intent, expected)  # type: ignore[union-attr]

    def test_similarity_confidence_is_bounded_and_available(self):
        match = self.engine.match("What is Python?")
        self.assertIsNotNone(match)
        self.assertGreaterEqual(match.confidence, self.engine.threshold)  # type: ignore[union-attr]
        self.assertLessEqual(match.confidence, 1.0)  # type: ignore[union-attr]
        self.assertTrue(match.example)  # type: ignore[union-attr]

    def test_typo_tolerance(self):
        match = self.engine.match("what is Pyhton used for")
        self.assertIsNotNone(match)
        self.assertEqual(match.intent, "python")  # type: ignore[union-attr]

    def test_empty_or_unrelated_input_does_not_match(self):
        self.assertIsNone(self.engine.match("   "))
        strict_engine = IntentEngine(self.engine.knowledge, threshold=0.95)
        self.assertIsNone(strict_engine.match("Explain the weather forecast for tomorrow"))

    def test_threshold_validation(self):
        with self.assertRaises(ValueError):
            IntentEngine(self.engine.knowledge, threshold=1.1)


class ConversationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = IntentEngine(KnowledgeBase.load(ROOT / "knowledge_base" / "intents.json"))

    def setUp(self):
        self.session = ConversationSession(self.engine)

    def test_empty_input_is_handled_without_crashing(self):
        self.assertEqual(self.session.reply("  "), EMPTY_MESSAGE)

    def test_history_records_turns_and_repeated_messages(self):
        first = self.session.reply("What is Python?")
        second = self.session.reply("What is Python?")
        self.assertTrue(first)
        self.assertTrue(second)
        self.assertEqual(len(self.session.history), 4)
        self.assertEqual([turn.role for turn in self.session.history], ["user", "assistant", "user", "assistant"])

    def test_follow_up_uses_recent_context(self):
        self.session.reply("What is Python?")
        answer = self.session.reply("What is it used for?")
        self.assertIn("automation", answer.lower())
        self.assertEqual(self.session.history[-1].intent, "python")

    def test_clear_resets_history_and_context(self):
        self.session.reply("What is Python?")
        self.session.clear()
        self.assertEqual(self.session.history, [])
        self.assertIsNone(self.session.last_intent)

    def test_unknown_input_uses_relevant_fallback(self):
        answer = self.session.reply("What is the best recipe for a tropical dessert?")
        self.assertTrue(any(term in answer.lower() for term in ("not confident", "don't have a reliable match")))


class IntegrationTests(unittest.TestCase):
    def test_cli_starts_answers_and_exits(self):
        result = subprocess.run(
            [sys.executable, "chatbot.py"], cwd=ROOT, input="hello\nexit\n",
            text=True, capture_output=True, timeout=20,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Welcome to DecodeBot AI", result.stdout)
        self.assertIn("Goodbye", result.stdout)

    def test_original_cli_functionality_is_preserved(self):
        result = subprocess.run(
            [sys.executable, "chatbot_legacy.py"], cwd=ROOT,
            input="hello\nwhat is ai\nexit\n", text=True, capture_output=True, timeout=20,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Hello! How can I help you?", result.stdout)
        self.assertIn("AI stands for Artificial Intelligence.", result.stdout)

    def test_streamlit_app_renders_without_exceptions(self):
        try:
            from streamlit.testing.v1 import AppTest
        except ImportError as exc:  # dependencies are declared in requirements.txt
            self.fail(f"Streamlit testing support is unavailable: {exc}")
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=20).run()
        self.assertFalse(app.exception, "; ".join(str(error.message) for error in app.exception))


if __name__ == "__main__":
    unittest.main()
