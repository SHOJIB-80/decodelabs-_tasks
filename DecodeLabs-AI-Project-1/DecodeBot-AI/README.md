# DecodeBot AI

DecodeBot AI is a small, locally running educational chatbot that upgrades the original DecodeLabs internship demo. It uses text normalization, word/character TF-IDF features, cosine similarity, fuzzy string similarity, and inspectable phrase cues to select an intent from a JSON knowledge base. The response is chosen from locally stored text; the program does **not** use a large language model or generate unrestricted answers.

## Features

- Offline intent matching after dependencies are installed; no API key or external service is used.
- Intents, example utterances, responses, and selected follow-up responses stored in `knowledge_base/intents.json`.
- Confidence threshold and useful fallback for unsupported questions.
- Session-scoped conversation history, a clear-chat command/button, and narrowly scoped follow-up handling (for example, asking what Python is used for after asking what Python is).
- Command-line chat and a Streamlit dashboard with suggested prompts.
- The original demo is retained as `chatbot_legacy.py`; its screenshot is retained under `assets/`.
- Unit tests and a separate held-out evaluation set. Evaluation prompts are distinct from the knowledge-base training examples.

## Requirements

- Python 3.10 or newer (tested in the development environment with Python 3.12).
- Internet access only to install the Python packages initially. Chat inference and the knowledge base are local once installation is complete.

## Windows PowerShell setup

1. Extract `DecodeBot-AI.zip` to a folder.
2. In VS Code, choose **File → Open Folder…** and open the extracted `DecodeBot-AI` folder. Open a PowerShell terminal in VS Code.
3. Create and activate a virtual environment:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   If PowerShell blocks activation, you may instead permit locally created scripts for your user with `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`, then activate again. You can also avoid activation and use `.\.venv\Scripts\python.exe` for the commands below.

4. Install dependencies:

   ```powershell
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

## Run it

Start the terminal chatbot:

```powershell
python chatbot.py
```

Type `exit` or `quit` to leave; type `/clear` to reset conversation history. The original rule-based demo remains runnable as:

```powershell
python chatbot_legacy.py
```

Start the dashboard:

```powershell
python -m streamlit run app.py
```

Streamlit will print a local URL, typically `http://localhost:8501`, which you can open in a browser. Use **Clear conversation** in the sidebar to reset the current session.

## Test and evaluate

Run automated tests:

```powershell
python -m unittest discover -s tests -v
```

Run the separate held-out intent evaluation:

```powershell
python evaluate.py
```

The evaluation reports accuracy plus macro precision, recall, and F1. The held-out prompts in `tests/evaluation_set.json` are not used as training examples. These scores measure this small fixed intent set only; they do not demonstrate broad language understanding.

## Extend the knowledge base

Add an intent object in `knowledge_base/intents.json` with a unique `name`, at least two varied `examples`, and one or more `responses`. For contextual follow-ups, add a concise `follow_up` response to the relevant intent. Keep examples representative but do not simply duplicate one sentence with different punctuation. The matching model is rebuilt from this local file at application startup.

The default confidence threshold is `0.48`. This ranking score is not a calibrated probability. For the CLI it can be changed with, for example, `python chatbot.py --threshold 0.50`. Raising it tends to reduce unrelated responses but may increase fallbacks; lowering it can increase false matches. Phrase cues in the JSON are a small, transparent aid for distinguishing named topics and common conversation acts. The Streamlit threshold uses the same default, configured in `nlp_engine.py`.

## Project layout

```text
DecodeBot-AI/
├── app.py                       # Streamlit dashboard
├── chatbot.py                   # Current CLI
├── chatbot_legacy.py            # Preserved original exact-match CLI
├── conversation.py              # Session history and follow-up behavior
├── nlp_engine.py                # Normalization, TF-IDF, cosine and fuzzy matching
├── evaluate.py                  # Held-out evaluation and metrics
├── knowledge_base/intents.json  # Local intents, examples, responses
├── tests/                       # unittest suite and held-out prompts
├── assets/                      # Preserved original screenshot
├── requirements.txt
└── README.md
```

## Limitations

- TF-IDF and fuzzy similarity are lightweight text-matching techniques, not a large language model. They cannot reliably understand every paraphrase, maintain deep context, or answer arbitrary factual questions.
- Conversation context is limited to the current CLI or Streamlit session and a small set of explicit follow-up patterns. It is not saved to disk and does not carry across sessions.
- The knowledge base is intentionally small and educational. An unknown question should receive a fallback; do not treat this tool as a source for high-stakes advice.
- The held-out set is small, so the reported metrics are a diagnostic rather than a statistically strong benchmark. Add representative unseen evaluation prompts whenever the knowledge base changes.
- Dependencies must be installed before offline use. Updating them or accessing package indexes requires internet access.
