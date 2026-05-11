# AI Finance Assistant (course-style MVP)

Multi-agent financial **education** stack built with **LangGraph**, **LangChain**, **sentence-transformers + FAISS RAG**, **yfinance** quotes, and a **Streamlit** UI.

> **Reminder:** Outputs are illustrative teaching materials—not personalized investment or tax guidance.

## Project layout

```text
ai_finance_assistant/
├── src/ai_finance_assistant/
│   ├── agents/          # prompts + portfolio helpers
│   ├── core/            # LLM factories, YAML settings, logging
│   ├── rag/             # ingestion, embeddings, FAISS build + retrieval
│   ├── utils/           # market client (yfinance + optional Alpha Vantage)
│   ├── web_app/         # Streamlit entrypoint
│   └── workflow/        # LangGraph router + six specialist nodes
├── scripts/
│   └── materialize_kb.py  # generates ~50 markdown curriculum files
├── tests/
├── config.yaml
├── pyproject.toml
└── README.md
```

## Quick start

Open **PowerShell** or **Command Prompt**.

**1 — Go to the project folder** (use the folder that contains `pyproject.toml` and `README.md`):

- If your repo is `finnie-finance` and this app lives inside it:

  ```powershell
  cd c:\Users\aurbi\projects\finnie-finance\ai_finance_assistant
  ```

  If you copied the project elsewhere, `cd` to **that** path instead.

**2 — Create a virtual environment** (recommended on Windows):

```powershell
python -m venv .venv
```

If `python` is not recognized, try:

```powershell
py -3 -m venv .venv
```

**3 — Activate the venv**

PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

If you see *“cannot be loaded because running scripts is disabled”*, run **once**:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then try activating again.  
In **cmd.exe** instead:

```cmd
.\.venv\Scripts\activate.bat
```

**4 — Install dependencies**

```powershell
pip install -e ".[dev]"
```

**5 — Configure and prepare data**

```powershell
copy .env.example .env
# Edit .env: set OPENAI_API_KEY (or the key for whichever llm.provider you use in config.yaml)
python scripts\materialize_kb.py
python -m ai_finance_assistant.rag.build_index
```

If `python -m ai_finance_assistant...` fails with “No module named ai_finance_assistant”, reinstall step 4 in this same terminal, or temporarily:

```powershell
$env:PYTHONPATH = "src"
python -m ai_finance_assistant.rag.build_index
Remove-Item Env:PYTHONPATH
```

**6 — Run Streamlit**

```powershell
streamlit run src\ai_finance_assistant\web_app\app.py
```

### Environment variables

| Variable | Purpose |
| --- | --- |
| `OPENAI_API_KEY` | Default chat model provider (`config.yaml` → `llm.provider: openai`) |
| `GOOGLE_API_KEY` | Switch `llm.provider` to `google_genai` for Gemini |
| `ANTHROPIC_API_KEY` | Switch `llm.provider` to `anthropic` |
| `ALPHA_VANTAGE_API_KEY` | Optional backup quote source with rate-limit handling |

## LangGraph agents

1. `finance_qa` – general literacy with broad KB retrieval  
2. `portfolio_analysis` – blends computed allocation stats with portfolio-flavored excerpts  
3. `market_analysis` – pairs yfinance snapshots with contextual articles  
4. `goal_planning` – conversational planning guardrails anchored to Goals tab inputs  
5. `news_synthesis` – neutral synthesis prompts over finance + behavioral excerpts  
6. `tax_education` – taxed KB category filter with stronger disclaimers  

Routing lives in [`workflow/router.py`](src/ai_finance_assistant/workflow/router.py). Multi-turn chats use LangGraph’s `MemorySaver` keyed per Streamlit thread.

## RAG & knowledge maintenance

Articles live in `data/articles/*.md`. Regenerate stubs + rebuild the index anytime:

```powershell
python scripts\materialize_kb.py
python -m ai_finance_assistant.rag.build_index
```

Metadata such as embedding model snapshot is emitted to `data/index/metadata.json`.

## Testing

```powershell
pytest --cov=ai_finance_assistant --cov-report=term-missing
```

Unit tests deliberately avoid outbound LLM/market dependencies; extend with integration mocks as needed.

## Troubleshooting

- **`ConfigurationError` on launch** → ensure `.env` has the provider key referenced in [`config.yaml`](config.yaml).
- **Partial `pip install -e .` (WinError 32 on sklearn)** → another process is locking site-packages; close IDEs using that Python, then retry, or use a fresh venv under `ai_finance_assistant/.venv`.
- **`ModuleNotFoundError: ai_finance_assistant` after a failed install** → run with `PYTHONPATH=src` (PowerShell: `$env:PYTHONPATH="src"`) or fix the editable install.
- **“Vector store not loaded” hints** → run `scripts/materialize_kb.py` then rebuild the index.
- **Thin RAG hits** → ensure article `category` matches the filters in retrieval calls (`investing_basics`, `portfolio`, `markets`, `retirement`, `taxes`, `behavioral`).
- **Upstream API flakiness** → yfinance intermittently blanks; retries + Alpha Vantage toggle help but never replace human verification before decisions.
