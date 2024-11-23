from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings


def select_emebddings(backend='ollama'):
    """
    Selects the OpenAI embeddings.
    Returns:
        OpenAIEmbeddings: The OpenAI embeddings.
    """
    if backend == 'ollama':
        embeddings = OllamaEmbeddings()
    
    elif backend == 'openai':
        embeddings = OpenAIEmbeddings()
    return embeddings


def create_faiss_vector(documents):
    """
    Creates a FAISS vector from the given documents.
    Args:
        documents: The documents from which the FAISS vector will be created.
    Returns:
        FAISS: The FAISS vector created from the given documents.
    """
    embeddings = OpenAIEmbeddings()
    vector = FAISS.from_documents(documents, embeddings)
    return vector