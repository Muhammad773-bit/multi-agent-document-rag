import os
from langchain.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader, UnstructuredPowerPointLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from app.utils import DATA_DIR, INDEX_DIR

class IngestionAgent:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    def ingest_from_dir(self):
        docs = []
        for file in os.listdir(DATA_DIR):
            path = os.path.join(DATA_DIR, file)

            if file.endswith(".pdf"):
                docs += PyPDFLoader(path).load()
            elif file.endswith(".docx"):
                docs += UnstructuredWordDocumentLoader(path).load()
            elif file.endswith(".pptx"):
                docs += UnstructuredPowerPointLoader(path).load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
        docs = splitter.split_documents(docs)

        vectorstore = FAISS.from_documents(docs, self.embeddings)
        vectorstore.save_local(INDEX_DIR)
        return "Ingested and indexed all files from directory!"

    async def ingest_uploaded_file(self, uploaded_file):
        path = os.path.join(DATA_DIR, uploaded_file.filename)
        with open(path, "wb") as f:
            f.write(await uploaded_file.read())

        docs = []
        if path.endswith(".pdf"):
            docs += PyPDFLoader(path).load()
        elif path.endswith(".docx"):
            docs += UnstructuredWordDocumentLoader(path).load()
        elif path.endswith(".pptx"):
            docs += UnstructuredPowerPointLoader(path).load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
        docs = splitter.split_documents(docs)

        vectorstore = FAISS.from_documents(docs, self.embeddings)
        vectorstore.save_local(INDEX_DIR)
        return "Uploaded file indexed successfully!"
