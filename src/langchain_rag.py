from langchain.tools.retriever import create_retriever_tool
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.agents import create_openai_functions_agent
from langchain.agents import AgentExecutor
from langchain_core.messages import HumanMessage, AIMessage

from dataclasses import dataclass


@dataclass
class RAGConfig:
    link: str = "https://docs.smith.langchain.com/user_guide"


class RAGAgent:
    # TODO:
    # 1. USE RAGConfig to set the link
    # 2. Add llm as a parameter to the constructor, as of now openai is hardcoded
    def __init__(self, link="https://docs.smith.langchain.com/user_guide"):
        self.link = link
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    def get_docs(self):
        loader = WebBaseLoader(self.link)
        docs = loader.load()
        return docs

    def split_docs(self, docs):
        text_splitter = RecursiveCharacterTextSplitter()
        documents = text_splitter.split_documents(docs)
        return documents

    def get_vector(self, documents):
        embeddings = OpenAIEmbeddings()
        vector = FAISS.from_documents(documents, embeddings)
        return vector

    def pipeline(self):
        docs = self.get_docs()
        documents = self.split_docs(docs)
        vector = self.get_vector(documents)
        retriever = vector.as_retriever()
        return retriever

    def create_agent_with_search(
        self, user_input="how can langsmith help with testing?"
    ):
        retriever = self.pipeline()  # self.get_retriever()
        retriever_tool = create_retriever_tool(
            retriever,
            "langsmith_search",
            "Search for information about LangSmith. For any questions about LangSmith, you must use this tool!",
        )

        search = TavilySearchResults()
        tools = [retriever_tool, search]
        prompt = hub.pull("hwchase17/openai-functions-agent")
        agent = create_openai_functions_agent(self.llm, tools, prompt)
        self.agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    def chat_with_agent(self, message="Can LangSmith help test my LLM applications?"):
        self.create_agent_with_search()
        self.agent_executor = self.get_agent()
        chat_history = [
            HumanMessage(content=message),
            AIMessage(content="Yes!"),
        ]
        invoke_arg = {"chat_history": chat_history, "input": "Tell me how"}
        response = self.agent_executor.invoke(invoke_arg)
        print(response["output"])

    def get_agent(self):
        return self.agent_executor


if __name__ == "__main__":
    rag_agent = RAGAgent()
    rag_agent.chat_with_agent(message="what is the weather in el salvador")
