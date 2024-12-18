from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

from ai_assistant.models.get_llm import get_llm


@tool
def magic_function(input: int) -> int:
    """
    Adds 2 to the given input integer.
    Args:
        input (int): The integer to which 2 will be added.
    Returns:
        int: The result of adding 2 to the input integer.
    """
    return int(input) + 2


def agent_with_tools(model, tools, query):
    """
    Creates a ReAct agent with the specified model and tools, and processes the given query.
    Args:
        model: The model to be used by the ReAct agent.
        tools: A list of tools to be used by the ReAct agent.
        query: The input query to be processed by the agent.
    Returns:
        dict: A dictionary containing the input query and the agent's response.
    """

    graph = create_react_agent(model, tools)
    messages = graph.invoke({"messages": [("human", query)]})
    return {"input": query, "output": messages["messages"][-1].content}


if __name__ == "__main__":
    create_react_agent()
