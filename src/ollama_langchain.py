import os
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


class LLMChain:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.llm = Ollama(model="llama2")

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


if __name__ == "__main__":
    chat = LLMChain()
    chat.answer()
