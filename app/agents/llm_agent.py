from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from app.config import HF_TOKEN

class LLMAgent:
    def __init__(self):
        self.llm = HuggingFaceHub(
            repo_id="google/flan-t5-small",
            huggingfacehub_api_token=HF_TOKEN,
            task="text2text-generation",
            model_kwargs={"temperature": 0, "max_new_tokens": 256}
        )

    def answer(self, retriever, query):
        qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=retriever,
            chain_type="stuff"
        )
        return qa.run(query)
    