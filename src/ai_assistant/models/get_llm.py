from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama


def get_llm(llm_type: str = ""):
    """
    Retrieve a language model instance based on the specified type.
    Parameters:
    llm_type (str): The type of language model to retrieve.
                    Options are "ollama" (default) and "openai".
    Returns:
    model: An instance of the specified language model.
    Raises:
    ValueError: If the specified llm_type is not recognized.
    """

    if llm_type in ("llama3.1", "ollama"):
        model = ChatOllama(model="llama3.1")

    elif llm_type == "llama3.2":
        model = ChatOllama(model="llama3.2")
    elif llm_type in ("openai", "gpt-4o-mini"):
        model = ChatOpenAI(model="gpt-4o-mini")
    else:
        raise ValueError(f"Model {llm_type} not recognized")
    return model
