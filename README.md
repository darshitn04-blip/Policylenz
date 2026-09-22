# PolicyLens

A simple student RAG project that explains Indian government-scheme documents in plain English, shows source citations, and runs transparent eligibility pre-checks.

## What it demonstrates

- **Messy-PDF ingestion:** `pypdf` extracts text page-by-page and a defensive overlapping chunker handles uneven formatting.
- **RAG:** SentenceTransformer embeddings + a local FAISS vector index retrieve relevant source chunks.
- **Citations:** every retrieved passage includes document name and page number for uploaded PDFs.
- **Structured logic:** a small, inspectable Python decision table handles eligibility pre-checks instead of asking an LLM to make the decision.
- **Evals:** a JSON Q&A set and runner make accuracy measurable. Expand it to 15–20 manually verified examples before using the resume metric.

## Run locally

```powershell
cd "C:\Users\Dell\Documents\New project 2\policylens"
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

The first full install/run downloads the compact `all-MiniLM-L6-v2` embedding model. If those heavier dependencies are unavailable, the app transparently uses a local TF-IDF fallback; install the listed requirements to use semantic SentenceTransformer + FAISS retrieval. The app works without an LLM API key: it gives cited extractive evidence. To enable concise generated answers, set `OPENAI_API_KEY` in your environment (and optionally `POLICYLENS_MODEL`).

## Add real official documents

Use the sidebar to upload official PDFs. For a portfolio version, download and retain the source URLs and publication dates in a `sources.md` file. Do not call this an official government service, and keep the on-screen final-eligibility disclaimer.

## Evaluate

```powershell
python evals/run_evals.py
python -m pytest
```

Add human-verified cases to `evals/qa_set.json`; the evaluator reports simple must-contain accuracy. Record the test date, document versions, and failures in your project report.
