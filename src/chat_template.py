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


class ChatTemplate:

    def __init__(
        self,
        llm_backend="ollama",
        openai_model="gpt-3.5-turbo-0125",
        ollama_model="llama2",
    ):
        self.accepted_llm_backends = [""]
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.llm_backend = llm_backend
        self.openai_model = openai_model
        self.ollama_model = ollama_model

    def init_llm(self):
        try:
            if self.llm_backend == "ollama":
                self.llm = Ollama(model=self.ollama_model)

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

    def question(self, input="How can langsmith help with testing?"):
        output_parser = StrOutputParser()

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are world class technical documentation writer."),
                ("user", "{input}"),
            ]
        )
        chain = prompt | self.llm | output_parser
        print(chain.invoke(prompt))

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

    def retriaval(self):
        pass


if __name__ == "__main__":
    chat = ChatPromptTemplate(llm_backend="ollama", ollama_model="llama2")
    chat.question()
