"""Conversation state and narrowly scoped follow-up handling."""
from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Any

from nlp_engine import IntentEngine, normalize_text

EMPTY_MESSAGE = "Please enter a message before sending."
FALLBACK_RESPONSES = (
    "I'm not confident I understood that. Try asking about Python, programming, AI, or machine learning.",
    "I don't have a reliable match for that yet. Try rephrasing it or choose one of the suggested questions.",
)
_FOLLOW_UP_PATTERNS = (
    r"what is it used for", r"what are they used for", r"what can it do",
    r"how does it work", r"tell me more", r"give me an example",
    r"what are its uses", r"and what about its uses", r"why is it useful",
)


@dataclass
class Turn:
    role: str
    content: str
    intent: str | None = None
    confidence: float | None = None

    def as_dict(self) -> dict[str, Any]:
        result = {"role": self.role, "content": self.content}
        if self.intent is not None:
            result["intent"] = self.intent
        if self.confidence is not None:
            result["confidence"] = self.confidence
        return result


class ConversationSession:
    """One chat session; the caller (CLI or Streamlit) owns its lifetime."""

    def __init__(self, engine: IntentEngine, fallback_threshold: float | None = None):
        self.engine = engine
        self.history: list[Turn] = []
        self.last_intent: str | None = None
        self.fallback_threshold = engine.threshold if fallback_threshold is None else fallback_threshold

    def clear(self) -> None:
        self.history.clear()
        self.last_intent = None

    def reply(self, text: str) -> str:
        if not isinstance(text, str) or not text.strip():
            return EMPTY_MESSAGE
        normalized = normalize_text(text)
        self.history.append(Turn("user", text.strip()))
        follow_up = self._is_follow_up(normalized)
        if follow_up and self.last_intent:
            follow_response = self.engine.knowledge.intents[self.last_intent].get("follow_up")
            if isinstance(follow_response, str) and follow_response.strip():
                answer = follow_response.strip()
                self.history.append(Turn("assistant", answer, self.last_intent, 1.0))
                return answer
        match = self.engine.match(text)
        if match is None or match.confidence < self.fallback_threshold:
            answer = random.choice(FALLBACK_RESPONSES)
            self.history.append(Turn("assistant", answer, None, match.confidence if match else None))
            return answer
        self.last_intent = match.intent
        answer = self.engine.response_for(match.intent)
        self.history.append(Turn("assistant", answer, match.intent, match.confidence))
        return answer

    @staticmethod
    def _is_follow_up(normalized: str) -> bool:
        return any(re.fullmatch(pattern, normalized) for pattern in _FOLLOW_UP_PATTERNS)

    def history_as_dicts(self) -> list[dict[str, Any]]:
        return [turn.as_dict() for turn in self.history]
