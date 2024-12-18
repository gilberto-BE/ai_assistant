#!/usr/bin/env python
from typing import List

from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain import hub
from langchain.agents import create_openai_functions_agent
from langchain.agents import AgentExecutor
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
from langserve import add_routes
from ai_assistant.langchain_code.rag import RAGAgent


# def define_app():
#     app = FastAPI(
#         title="LangChain Server",
#         version="1.0",
#         description="A simple API server using LangChain's Runnable interfaces",
#     )
#     return app


# app = define_app()


# @app.get("/test")
# def test_route():
#     return {"message": "This is a test route"}


# class Input(BaseModel):
#     input: str
#     chat_history: List[BaseMessage] = Field(
#         ...,
#         extra={"widget": {"type": "chat", "input": "location"}},
#     )


# class Output(BaseModel):
#     output: str


# rag_agent = RAGAgent()
# rag_agent.chat_with_agent(message="what is the weather in el salvador")
# agent_executor = rag_agent.get_agent()
# add_routes(
#     app,
#     agent_executor.with_types(input_type=Input, output_type=Output),
#     path="/agent",
# )

# if __name__ == "__main__":
#     import uvicorn

#     uvicorn.run(app, host="localhost", port=8000)


#!/usr/bin/env python
from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatAnthropic, ChatOpenAI
from langserve import add_routes

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple api server using Langchain's Runnable interfaces",
)

add_routes(
    app,
    ChatOpenAI(model="gpt-3.5-turbo-0125"),
    path="/openai",
)

add_routes(
    app,
    ChatAnthropic(model="claude-3-haiku-20240307"),
    path="/anthropic",
)

model = ChatAnthropic(model="claude-3-haiku-20240307")
prompt = ChatPromptTemplate.from_template("tell me a joke about {topic}")
add_routes(
    app,
    prompt | model,
    path="/joke",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
