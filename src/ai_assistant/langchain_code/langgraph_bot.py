from typing import TypedDict

from regex import T
from typing import Annotated
from langgraph.graph import StateGraph, START, END

from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from dataclasses import dataclass, field
from langchain_ollama import ChatOllama


class State(TypedDict):
    messages: Annotated[list, add_messages]


class ChatBot:
    def __init__(self, model: str):
        self.llm = ChatOpenAI(model=model)

    def chat(self, state: State):
        return {"messages": self.llm.invoke(state["messages"])}


# def chatbot():
#     bot = ChatBot(model="gpt-4o-mini")
#     return bot.chat()

# llm = ChatOpenAI(model="gpt-4o-mini")

llm = ChatOllama(model="llama3.1")


def chatbot(state: State):
    return {"messages": llm.invoke(state["messages"])}


def create_graph(func=chatbot):
    graph_builder = StateGraph(State)
    graph_builder.add_node("chatbot", func)
    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_edge("chatbot", END)
    graph = graph_builder.compile()
    return graph


def stream_graph_update(user_input: str):
    for event in graph.stream({"messages": [("user", user_input)]}):
        for value in event.values():
            # print(f"Value['messages']: {value['messages']}")
            print("Assistant:", value["messages"].content)


if __name__ == "__main__":
    graph = create_graph()

    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            stream_graph_update(user_input)
        except Exception as e:
            user_input = "What do you know about LangGraph?"
            print("User: ", user_input)
            stream_graph_update(user_input)
            break
