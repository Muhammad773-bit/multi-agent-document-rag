from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from app.utils import INDEX_DIR

class RetrievalAgent:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    def get_retriever(self):
        vectorstore = FAISS.load_local(INDEX_DIR, self.embeddings)
        return vectorstore.as_retriever()

    def retrieve_docs(self, query):
        retriever = self.get_retriever()
        return retriever.get_relevant_documents(query)
