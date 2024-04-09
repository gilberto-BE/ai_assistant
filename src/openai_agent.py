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


class RagAgent:

    def __init__(self, link="https://docs.smith.langchain.com/user_guide"):
        self.link = link
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    def load_homepage(
        self,
    ):
        loader = WebBaseLoader(self.link)
        docs = loader.load()
        embeddings = OpenAIEmbeddings()

        text_splitter = RecursiveCharacterTextSplitter()
        documents = text_splitter.split_documents(docs)
        vector = FAISS.from_documents(documents, embeddings)
        return vector

    def get_retriever(self):
        vector = self.load_homepage()
        retriever = vector.as_retriever()
        return retriever

    def agent_search(self, user_input="how can langsmith help with testing?"):
        retriever = self.get_retriever()
        retriever_tool = create_retriever_tool(
            retriever,
            "langsmith_search",
            "Search for information about LangSmith. For any questions about LangSmith, you must use this tool!",
        )

        search = TavilySearchResults()
        tools = [retriever_tool, search]
        prompt = hub.pull("hwchase17/openai-functions-agent")
        agent = create_openai_functions_agent(self.llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
        # invoke_arg = {"input": user_input}
        # response = agent_executor.invoke(invoke_arg)
        # print(response["output"])
        chat_history = [
            HumanMessage(content="Can LangSmith help test my LLM applications?"),
            AIMessage(content="Yes!"),
        ]
        invoke_arg = {"chat_history": chat_history, "input": "Tell me how"}
        response = agent_executor.invoke(invoke_arg)
        print(response["output"])


if __name__ == "__main__":
    rag_agent = RagAgent()
    rag_agent.agent_search(
        "how did the NYSE and stockholm stock marekets closed today? Please give the answers in percentage change."
    )
