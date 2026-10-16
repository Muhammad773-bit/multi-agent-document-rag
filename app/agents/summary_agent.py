from langchain.chains.summarize import load_summarize_chain
from app.agents.retrieval_agent import RetrievalAgent
from langchain.llms import HuggingFaceHub
from app.config import HF_TOKEN

class SummaryAgent:
    def __init__(self):
        self.retrieval_agent = RetrievalAgent()
        self.llm = HuggingFaceHub(
            repo_id="google/flan-t5-small",
            huggingfacehub_api_token=HF_TOKEN,
            task="text2text-generation",
            model_kwargs={"temperature": 0, "max_new_tokens": 256}
        )

    def summarize(self):
        retriever = self.retrieval_agent.get_retriever()
        docs = retriever.get_relevant_documents("summarize the documents")
        chain = load_summarize_chain(self.llm, chain_type="map_reduce")
        return chain.run(docs)
