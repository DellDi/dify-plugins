import chromadb.utils.embedding_functions as embedding_functions
import os
from dotenv import load_dotenv

load_dotenv()

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv("EMBEDDING_API_KEY"),
    api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_type="tongyi",
    api_version="v3",
    model_name="text-embedding-v2",
)
