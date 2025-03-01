import faiss
import numpy as np
from typing import Tuple, List
import pandas as pd
from langchain.embeddings import OpenAIEmbeddings
from openai import OpenAI
from config import *

def create_embeddings(df: pd.DataFrame, column_name: str, model: str) -> np.ndarray:
    """
    This function loads the OpenAI embedding model, encodes the text data in the specified column, 
    and returns a NumPy array of the embeddings.

    Args:
        df (pandas.DataFrame): The DataFrame containing the text data.
        column_name (str): The name of the column containing the text data.
        model (str): The name of the OpenAI embedding model.

    Returns:
        np.ndarray: A NumPy array containing the vector embeddings.
    """
    embeddings = OpenAIEmbeddings(openai_api_key=API_KEY, model=model)
    #encode the text data in the specified column using the sentence transformer model
    df[f"{column_name}_vector"] = df[column_name].apply(lambda x: embeddings.embed_query(x))
    #stack the encoded vectors into a NumPy array
    vectors = np.stack(df[f"{column_name}_vector"].values)
    
    return vectors

def create_or_update_index(vectors: np.ndarray, index_file_path: str) -> faiss.Index:
    """
    Creates or updates a FAISS index while supporting different distance metrics (L2 or Cosine Similarity).

    Args:
        vectors (np.ndarray): A NumPy array containing the vector embeddings.
        index_file_path (str): The path to save or update the FAISS index file.

    Returns:
        faiss.Index: The updated FAISS index.
    """
    dimension = vectors.shape[1]  # Get embedding dimension

    # Choose FAISS distance metric based on config
    if FAISS_DISTANCE_METRIC == "L2":
        index_type = faiss.IndexFlatL2(dimension)
        print("Using L2 (Euclidean Distance) for FAISS index.")
    elif FAISS_DISTANCE_METRIC == "COSINE":
        index_type = faiss.IndexFlatIP(dimension)  # Inner Product (for Cosine Similarity)
        print("Using Cosine Similarity for FAISS index.")
    else:
        raise ValueError("Invalid FAISS_DISTANCE_METRIC. Use 'L2' or 'COSINE'.")

    # Check if index file exists
    if os.path.exists(index_file_path):
        print("Updating existing FAISS index...")
        index = faiss.read_index(index_file_path)
    else:
        print("Creating a new FAISS index...")
        index = index_type

    # Add new vectors to the index
    index.add(vectors)

    # Save the updated index
    faiss.write_index(index, index_file_path)
    print(f"FAISS index updated and saved at {index_file_path}")

    return index

def semantic_similarity(query: str, index: faiss.Index, model: str, k: int = DEFAULT_K) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates the semantic similarity between a query and a set of indexed vectors.

    Args:
        query (str): The query string.
        index (faiss.Index): The FAISS index used for searching.
        model (str): The name of the OpenAI embedding model used to create embedding.
        k (int, optional): The number of most similar vectors to retrieve. Defaults to 3.

    Returns:
        tuple: A tuple containing two arrays - D and I.
            - D (numpy.ndarray): The distances between the query vector and the indexed vectors.
            - I (numpy.ndarray): The indices of the most similar vectors in the index.
    """
    model = OpenAIEmbeddings(openai_api_key=API_KEY, model=model)
    #embed the query
    query_vector = model.embed_query(query)
    query_vector = np.array([query_vector]).astype('float32')
    #search the FAISS index
    D, I = index.search(query_vector, k)
    
    return D, I

def generate_response(query: str, responses: List[str], chat_history: str, model=CHAT_MODEL) -> str:
    """
    Calls the Language Model to generate a response based on the given query, list of responses, and chat history.

    Args:
        query (str): The customer query.
        responses (List[str]): A list of example responses from the internal database.
        chat_history (str): The conversation history.

    Returns:
        str: The generated response from the Language Model.
    """
    client = OpenAI(api_key=API_KEY)

    final_prompt = QUERY_PROMPT.format(query=query, responses=responses, chat_history=chat_history)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": final_prompt}
    ]
    response = client.chat.completions.create(model=model, messages=messages, temperature=TEMPERATURE)

    return response.choices[0].message.content

