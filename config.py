import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Configuration
API_KEY = os.getenv("OPENAI_API_KEY", "your_default_api_key")
CHAT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
EMBEDDING_MODEL= os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")

# FAISS Configuration
FAISS_DISTANCE_METRIC = os.getenv("FAISS_DISTANCE_METRIC", "L2")  # Options: "L2" or "Cosine"
INDEX_FILE_PATH = os.getenv("INDEX_FILE_PATH", "faiss_index.idx")
# Query Configuration
DEFAULT_K = int(os.getenv("DEFAULT_K", 3))

# LLM Configuration
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "You are a helpful assistant and help answer customer's query and request. Answer on the basis of provided context only.")
QUERY_PROMPT=os.getenv("QUERY_PROMPT","")
TEMPERATURE = float(os.getenv("TEMPERATURE", 0))  # 0 for deterministic, 1 for more randomness
