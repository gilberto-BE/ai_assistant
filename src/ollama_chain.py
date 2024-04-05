import os
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


class OpenAIChain:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.llm = Ollama(model="llama2")

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


if __name__ == "__main__":
    chat = OpenAIChain()
    chat.question()
