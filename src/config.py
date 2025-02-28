import os
from dotenv import load_dotenv
load_dotenv()

# API Configuration
# Read API key from environment (without a default)
API_KEY=os.getenv("OPENAI_API_KEY")

# Raise an error if API key is missing
if API_KEY is None:
    raise ValueError("ERROR: OPENAI_API_KEY is not set! Add it to your .env file or system environment variables.")
CHAT_MODEL = "gpt-4o"
EMBEDDING_MODEL="text-embedding-3-small"


# FAISS Configuration
FAISS_DISTANCE_METRIC = "L2"  # Options: "L2" or "Cosine"
INDEX_FILE_PATH = "faiss_index.idx"
# Query Configuration
DEFAULT_K = int(3)

# Prompts
SYSTEM_PROMPT ="You are a helpful assistant and help answer customer's query and request. Answer on the basis of provided context only."
QUERY_PROMPT="""On the basis of the input customer query determine or suggest the following things about the input query: {query}:
1. Urgency of the query on a scale of 1-5 where 1 is least urgent and 5 is most urgent. Only tell me the number.
2. Categorize the input query into sales, product, operations etc. Only tell me the category.
3. Generate 1 best humble response to the input query which is similar to examples in the python list: {responses} from the internal database and is helpful to the customer.
If the input query from the customer is not clear then ask a follow-up question."""
TEMPERATURE=float(0)

#File paths
# Get the absolute path to the root directory
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INPUT_DATA_PATH = os.path.join(ROOT_DIR, "Data", "Customer_Support_Training_Dataset.csv")
VECTOR_STORE_DIR = os.path.join(ROOT_DIR, "vector_store")
VECTOR_STORE_FILE = os.path.join(VECTOR_STORE_DIR, "faiss_index.index")





