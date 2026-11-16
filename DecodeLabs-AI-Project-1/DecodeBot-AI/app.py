"""Streamlit dashboard for the offline DecodeBot AI chatbot."""
from __future__ import annotations

import streamlit as st

from conversation import ConversationSession
from nlp_engine import DEFAULT_KNOWLEDGE_PATH, DEFAULT_THRESHOLD, IntentEngine, KnowledgeBase, KnowledgeBaseError

st.set_page_config(page_title="DecodeBot AI", page_icon="D", layout="centered")
st.markdown("""
<style>
  .block-container {max-width: 900px; padding-top: 2.2rem;}
  .hero {padding: 1.2rem 1.4rem; border: 1px solid #29354b; border-radius: 18px;
         background: linear-gradient(135deg, #121c2d, #18243a); margin-bottom: 1rem;}
  .hero h1 {margin: 0 0 .35rem 0; letter-spacing: -.04em;}
  .hero p {margin: 0; color: #b8c5d9;}
  [data-testid="stChatMessage"] {border: 1px solid rgba(148,163,184,.12); border-radius: 16px;}
  div[data-testid="stChatInput"] {border-color: #41516c;}
  .small-note {color: #9eacc1; font-size: .88rem;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_engine() -> IntentEngine:
    return IntentEngine(KnowledgeBase.load(DEFAULT_KNOWLEDGE_PATH), threshold=DEFAULT_THRESHOLD)

try:
    engine = load_engine()
except (KnowledgeBaseError, RuntimeError, ValueError) as exc:
    st.error(f"DecodeBot could not load its local knowledge base: {exc}")
    st.stop()

if "conversation" not in st.session_state:
    st.session_state.conversation = ConversationSession(engine)
if "messages" not in st.session_state:
    st.session_state.messages = []

st.markdown("""
<div class="hero">
  <h1>DecodeBot AI</h1>
  <p>A local, intent-based learning chatbot for Python, programming, AI, and machine learning.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.subheader("About DecodeBot")
    st.write("This app uses TF-IDF similarity and fuzzy text matching against a small local knowledge base.")
    st.caption("It is not a large language model and does not send messages to an external API.")
    st.divider()
    st.subheader("Suggested questions")
    suggestions = ["What is Python?", "Explain AI simply", "What is machine learning?", "How do I start programming?"]
    selected_suggestion = None
    for index, question in enumerate(suggestions):
        if st.button(question, key=f"suggestion_{index}", use_container_width=True):
            selected_suggestion = question
    st.divider()
    if st.button("Clear conversation", type="secondary", use_container_width=True):
        st.session_state.conversation.clear()
        st.session_state.messages = []
        st.rerun()

if not st.session_state.messages:
    st.info("Welcome! Ask a question about Python, programming, artificial intelligence, or machine learning.")
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask DecodeBot a question…")
if selected_suggestion:
    prompt = selected_suggestion
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Checking the local knowledge base…"):
            answer = st.session_state.conversation.reply(prompt)
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
