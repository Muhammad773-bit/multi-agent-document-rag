import os
from dotenv import load_dotenv
load_dotenv()

HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
