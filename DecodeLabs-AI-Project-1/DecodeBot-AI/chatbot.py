"""DecodeBot AI command-line chat interface."""
from __future__ import annotations

import argparse

from conversation import ConversationSession
from nlp_engine import DEFAULT_KNOWLEDGE_PATH, DEFAULT_THRESHOLD, IntentEngine, KnowledgeBase, KnowledgeBaseError


def main() -> int:
    parser = argparse.ArgumentParser(description="Chat with DecodeBot AI locally.")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help="minimum intent confidence from 0 to 1 (default: 0.48)")
    parser.add_argument("--knowledge-base", default=str(DEFAULT_KNOWLEDGE_PATH),
                        help="path to the intents JSON file")
    args = parser.parse_args()
    try:
        engine = IntentEngine(KnowledgeBase.load(args.knowledge_base), threshold=args.threshold)
    except (KnowledgeBaseError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
        return 2

    session = ConversationSession(engine)
    print("=" * 42)
    print("       Welcome to DecodeBot AI")
    print("=" * 42)
    print("Ask about Python, programming, AI, or machine learning.")
    print("Type 'exit' to quit or '/clear' to clear chat history.")
    while True:
        try:
            user_input = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print("\nBot: Goodbye! Have a great day.")
            break
        if user_input.strip().lower() in {"exit", "quit"}:
            print("Bot: Goodbye! Have a great day.")
            break
        if user_input.strip().lower() == "/clear":
            session.clear()
            print("Bot: Conversation history cleared.")
            continue
        print("Bot:", session.reply(user_input))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
