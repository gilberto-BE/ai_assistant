import os
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
import logging

from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.embeddings import CohereEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage


# logging.getLogger()

logging.basicConfig(
    filename="app.log",  # File where logs will be written
    filemode="a",  # Append mode, so logs are added to the file; use 'w' for overwrite mode
    format="%(asctime)s - %(levelname)s - %(message)s",  # Format for the log messages
    level=logging.DEBUG,  # Logging level, e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL
)


class ChatAgent:

    def __init__(
        self,
        llm_backend="ollama",
        openai_model="gpt-3.5-turbo",
        ollama_model="llama2",
    ):
        self.accepted_llm_backends = [""]
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.llm_backend = llm_backend
        self.openai_model = openai_model
        self.ollama_model = ollama_model
        self.init_llm()
        logging.info(f"Using backend: {self.llm_backend}")

    def init_llm(self):
        try:
            if self.llm_backend == "ollama":
                self.llm = Ollama(model=self.ollama_model)
                logging.info(f"Model name: {self.ollama_model}")

            elif self.llm_backend == "openai":
                self.llm = ChatOpenAI(
                    openai_api_key=self.api_key, model=self.openai_model
                )
            elif self.llm_backend == "anthorpic":
                pass
            elif self.llm_backend == "cohere":
                pass

        except Exception as e:
            logging.error(
                f"llm_backend has to be one of: {self.accepted_llm_backends} | {e}"
            )

    def answer(self, input="How can langsmith help with testing?"):
        output_parser = StrOutputParser()

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are world class technical documentation writer."),
                ("user", "{input}"),
            ]
        )
        # prompt.format(input=input)
        chain = prompt | self.llm | output_parser
        prompt = {"input": f"{input}"}
        print(chain.invoke(prompt))

    def set_embedding(self):
        if self.llm_backend == "ollama":
            self.embeddings = OllamaEmbeddings()
        elif self.llm_backend == "openai":
            self.embeddings = OpenAIEmbeddings()
        elif self.llm_backend == "cohere":
            self.embeddings = CohereEmbeddings()

    def get_embeddings(self):
        return self.embeddings

    def load_homepage(
        self,
        link="https://www.hemnet.se/bostader?item_types%5B%5D=bostadsratt&expand_locations=1000&location_ids%5B%5D=17744",
    ):
        loader = WebBaseLoader(link)

        docs = loader.load()
        embeddings = OpenAIEmbeddings()

        text_splitter = RecursiveCharacterTextSplitter()
        documents = text_splitter.split_documents(docs)
        vector = FAISS.from_documents(documents, embeddings)
        return vector

    def answer_with_retriaval(self):

        prompt = ChatPromptTemplate.from_template(
            """Answer the following answer based only on the provided context:

        <context>
        {context}
        </context>

        Question: {input}"""
        )

        document_chain = create_stuff_documents_chain(self.llm, prompt)
        vector = self.load_homepage(link="https://docs.smith.langchain.com/user_guide")
        self.retriever = vector.as_retriever()
        retrieval_chain = create_retrieval_chain(self.retriever, document_chain)
        response = retrieval_chain.invoke(
            {"input": "how can langsmith help with testing?"}
        )
        print(response["answer"])

    def chat_conversation(self):

        from langchain.chains import create_history_aware_retriever
        from langchain_core.prompts import MessagesPlaceholder
        from langchain_core.messages import HumanMessage, AIMessage

        # First we need a prompt that we can pass into an LLM to generate this search query

        prompt = ChatPromptTemplate.from_messages(
            [
                MessagesPlaceholder(variable_name="chat_history"),
                ("user", "{input}"),
                (
                    "user",
                    "Given the above conversation, generate a search query to look up to get information relevant to the conversation",
                ),
            ]
        )
        vector = self.load_homepage(link="https://docs.smith.langchain.com/user_guide")
        retriever = vector.as_retriever()
        retriever_chain = create_history_aware_retriever(self.llm, retriever, prompt)

        chat_history = [
            HumanMessage(content="Can LangSmith help test my LLM applications?"),
            AIMessage(content="Yes!"),
        ]
        retriever_chain.invoke({"chat_history": chat_history, "input": "Tell me how"})
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Answer the user's questions based on the below context:\n\n{context}",
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                ("user", "{input}"),
            ]
        )
        document_chain = create_stuff_documents_chain(self.llm, prompt)

        retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)
        chat_history = [
            HumanMessage(content="Can LangSmith help test my LLM applications?"),
            AIMessage(content="Yes!"),
        ]
        print(
            retrieval_chain.invoke(
                {
                    "chat_history": chat_history,
                    "context": "Provide additional context here if necessary",
                    "input": "Tell me how",
                }
            )
        )


#     def chat_conversation(self):
#         prompt = ChatPromptTemplate.from_messages(
#             [
#                 (
#                     "system",
#                     "Answer the user's questions based on the below context:\n\n{context}",
#                 ),
#                 MessagesPlaceholder(variable_name="chat_history"),
#                 ("user", "{input}"),
#             ]
#         )
#         document_chain = create_stuff_documents_chain(self.llm, prompt)
#         # prompt = ChatPromptTemplate.from_messages(
#         #     [
#         #         MessagesPlaceholder(variable_name="chat_history"),
#         #         ("user", "{input}"),
#         #         (
#         #             "user",
#         #             "Given the above conversation, generate a search query to look up to get information relevant to the conversation",
#         #         ),
#         #     ]
#         # )
#         vector = self.load_homepage(link="https://docs.smith.langchain.com/user_guide")
#         retriever = vector.as_retriever()
#         retriever_chain = create_history_aware_retriever(
#             self.llm,
#             retriever,
#             prompt,
#         )
#         retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)
#         chat_history = [
#             HumanMessage(content="Can LangSmith help test my LLM applications?"),
#             AIMessage(content="Yes!"),
#         ]
#         print(
#             retrieval_chain.invoke(
#                 {
#                     "chat_history": chat_history,
#                     "input": "Tell me how",
#                 },
#             )
#         )

#         # print(
#         #     retriever_chain.invoke(
#         #         {
#         #             "chat_history": chat_history,
#         #             "input": "Tell me how",
#         #         }
#         #     )
#         # )


# #

if __name__ == "__main__":
    logging.info("Tests performed with ollama + llama2.")
    home_page = "https://docs.smith.langchain.com/user_guide"
    chat = ChatAgent(
        llm_backend="ollama",
        openai_model="gpt-3.5-turbo-0125",
        ollama_model="llama2",
    )
    # chat.answer()
    # chat.answer_with_retriaval()
    chat.chat_conversation()
