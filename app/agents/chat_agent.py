from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
from app.agents.retrieval_agent import RetrievalAgent
from app.config import HF_TOKEN

class ChatAgent:
    def __init__(self):
        self.retrieval_agent = RetrievalAgent()
        self.llm = HuggingFaceHub(
            repo_id="google/flan-t5-small",
            huggingfacehub_api_token=HF_TOKEN,
            task="text2text-generation",
            model_kwargs={"temperature": 0, "max_new_tokens": 256}
        )

    def chat(self, question, chat_history):
        retriever = self.retrieval_agent.get_retriever()
        chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever
        )
        return chain({"question": question, "chat_history": chat_history})
