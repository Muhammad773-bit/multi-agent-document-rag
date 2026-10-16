# Multi Agent Document Rag

Document question answering split across specialised agents: an ingestion agent, a retrieval agent, a summary agent and a chat agent, served through a FastAPI app.

**Tech:** Python, FastAPI, LangChain, vector search, multi-agent design

## Run it

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

API keys are read from environment variables. Copy `.env.example` to `.env` and fill in your own keys.

---
By [Muhammad Ali](https://muhammad773-bit.github.io), BS Artificial Intelligence, SMIU Karachi.
