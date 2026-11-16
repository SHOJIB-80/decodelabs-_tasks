"""Offline intent matching for DecodeBot AI; this is not a generative language model."""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError as exc:  # pragma: no cover - depends on the user's environment
    raise RuntimeError(
        "DecodeBot AI requires scikit-learn. Install the project dependencies with: "
        "python -m pip install -r requirements.txt"
    ) from exc

DEFAULT_KNOWLEDGE_PATH = Path(__file__).parent / "knowledge_base" / "intents.json"
DEFAULT_THRESHOLD = 0.48
_CONTRACTIONS = {
    "what's": "what is", "who's": "who is", "i'm": "i am", "you're": "you are",
    "it's": "it is", "that's": "that is", "doesn't": "does not", "don't": "do not",
    "can't": "cannot", "isn't": "is not", "how's": "how is", "let's": "let us",
}


def normalize_text(text: str) -> str:
    """Normalize a user message by folding Unicode, expanding common contractions,
    lowercasing, removing punctuation, and collapsing whitespace.
    """
    if not isinstance(text, str):
        raise TypeError("Message must be a string.")
    value = unicodedata.normalize("NFKC", text).strip().lower()
    for contraction, expanded in _CONTRACTIONS.items():
        value = re.sub(r"\b" + re.escape(contraction) + r"\b", expanded, value)
    value = re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE)
    return re.sub(r"\s+", " ", value).strip()


class KnowledgeBaseError(ValueError):
    """Raised when the JSON knowledge base is missing or malformed."""


@dataclass(frozen=True)
class IntentMatch:
    intent: str
    confidence: float
    example: str


class KnowledgeBase:
    def __init__(self, data: dict[str, Any]):
        intents = data.get("intents") if isinstance(data, dict) else None
        if not isinstance(intents, list) or not intents:
            raise KnowledgeBaseError("Knowledge base must contain a non-empty 'intents' list.")
        self.intents: dict[str, dict[str, Any]] = {}
        for item in intents:
            if not isinstance(item, dict):
                raise KnowledgeBaseError("Each intent must be a JSON object.")
            name = item.get("name")
            examples = item.get("examples")
            responses = item.get("responses")
            hints = item.get("hints", {})
            if not isinstance(name, str) or not name.strip():
                raise KnowledgeBaseError("Each intent needs a non-empty string name.")
            if name in self.intents:
                raise KnowledgeBaseError(f"Duplicate intent name: {name}")
            if (not isinstance(examples, list) or len(examples) < 2
                    or not all(isinstance(x, str) and normalize_text(x) for x in examples)):
                raise KnowledgeBaseError(f"Intent '{name}' needs at least two non-empty text examples.")
            if not isinstance(responses, list) or not responses or not all(
                    isinstance(x, str) and x.strip() for x in responses):
                raise KnowledgeBaseError(f"Intent '{name}' needs at least one non-empty response.")
            if not isinstance(hints, dict) or any(
                    not isinstance(k, str) or not normalize_text(k)
                    or not isinstance(v, (int, float)) or not 0 <= v <= 1
                    for k, v in hints.items()):
                raise KnowledgeBaseError(f"Intent '{name}' hints must map text cues to scores from 0 to 1.")
            item_copy = dict(item)
            item_copy["examples"] = [normalize_text(x) for x in examples]
            item_copy["hints"] = {normalize_text(k): float(v) for k, v in hints.items()}
            self.intents[name] = item_copy

    @classmethod
    def load(cls, path: str | Path = DEFAULT_KNOWLEDGE_PATH) -> "KnowledgeBase":
        path = Path(path)
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except FileNotFoundError as exc:
            raise KnowledgeBaseError(f"Knowledge-base file not found: {path}") from exc
        except json.JSONDecodeError as exc:
            raise KnowledgeBaseError(f"Invalid JSON in knowledge base {path}: {exc}") from exc
        except OSError as exc:
            raise KnowledgeBaseError(f"Could not read knowledge base {path}: {exc}") from exc
        return cls(data)


class IntentEngine:
    """Rank examples using TF-IDF cosine similarity, fuzzy scores, and curated cues.

    Cues are short, inspectable topic/phrase anchors in the JSON knowledge base;
    they disambiguate entities such as Python and ML when generic sentence words
    otherwise dominate a small training set. The resulting score is a ranking
    confidence, not a calibrated probability.
    """

    def __init__(self, knowledge: KnowledgeBase, threshold: float = DEFAULT_THRESHOLD):
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Confidence threshold must be between 0 and 1.")
        self.knowledge = knowledge
        self.threshold = threshold
        self.examples: list[str] = []
        self.example_intents: list[str] = []
        for name, item in knowledge.intents.items():
            for example in item["examples"]:
                self.examples.append(example)
                self.example_intents.append(name)
        self.word_vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        self.char_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True)
        self.word_matrix = self.word_vectorizer.fit_transform(self.examples)
        self.char_matrix = self.char_vectorizer.fit_transform(self.examples)

    def match(self, text: str) -> IntentMatch | None:
        normalized = normalize_text(text)
        if not normalized:
            return None
        word_vec = self.word_vectorizer.transform([normalized])
        char_vec = self.char_vectorizer.transform([normalized])
        word_scores = cosine_similarity(word_vec, self.word_matrix)[0]
        char_scores = cosine_similarity(char_vec, self.char_matrix)[0]
        best_by_intent: dict[str, tuple[float, str]] = {}
        for i, example in enumerate(self.examples):
            fuzzy = SequenceMatcher(None, normalized, example).ratio()
            lexical = 0.65 * float(word_scores[i]) + 0.35 * float(char_scores[i])
            score = 0.78 * lexical + 0.22 * fuzzy
            intent = self.example_intents[i]
            if intent not in best_by_intent or score > best_by_intent[intent][0]:
                best_by_intent[intent] = (score, example)

        # A cue only lifts a score when the normalized cue appears as a whole
        # phrase/token; the terms live in JSON and can be maintained without code.
        padded = f" {normalized} "
        for intent, item in self.knowledge.intents.items():
            for cue, cue_score in item["hints"].items():
                if f" {cue} " in padded and cue_score > best_by_intent[intent][0]:
                    best_by_intent[intent] = (cue_score, f"[cue: {cue}]")
        intent, (score, example) = max(best_by_intent.items(), key=lambda pair: pair[1][0])
        if score < self.threshold:
            return None
        return IntentMatch(intent=intent, confidence=round(score, 4), example=example)

    def response_for(self, intent: str, rng=None) -> str:
        """Return one response for an intent; injectable RNG keeps tests deterministic."""
        import random
        if intent not in self.knowledge.intents:
            raise KeyError(f"Unknown intent: {intent}")
        choices = self.knowledge.intents[intent]["responses"]
        chooser = rng or random
        return chooser.choice(choices)
