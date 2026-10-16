from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from app.utils import ensure_dirs
from app.agents.ingestion_agent import IngestionAgent
from app.agents.retrieval_agent import RetrievalAgent
from app.agents.llm_agent import LLMAgent
from app.agents.chat_agent import ChatAgent
from app.agents.summary_agent import SummaryAgent

app = FastAPI()
ensure_dirs()

ingestion_agent = IngestionAgent()
retrieval_agent = RetrievalAgent()
llm_agent = LLMAgent()
chat_agent = ChatAgent()
summary_agent = SummaryAgent()

class Query(BaseModel):
    question: str

class ChatRequest(BaseModel):
    question: str
    chat_history: list = []

@app.get("/")
def home():
    return {"message": "Document Retrieval Agent Running"}

@app.post("/ingest")
def ingest():
    result = ingestion_agent.ingest_from_dir()
    return {"status": "success", "message": result}

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    result = await ingestion_agent.ingest_uploaded_file(file)
    return JSONResponse({"status": "success", "message": result})

@app.post("/query")
def query(q: Query):
    retriever = retrieval_agent.get_retriever()
    answer = llm_agent.answer(retriever, q.question)
    return {"answer": answer}

@app.post("/chat")
def chat(req: ChatRequest):
    response = chat_agent.chat(req.question, req.chat_history)
    return response

@app.get("/summarize")
def summarize():
    summary = summary_agent.summarize()
    return {"summary": summary}
