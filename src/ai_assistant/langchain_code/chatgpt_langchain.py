import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


class OpenAIChain:

    def __init__(self, model="gpt-3.5-turbo"):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model = model
        self.llm = ChatOpenAI(openai_api_key=self.api_key, model=self.model)

    def answer(self, input="How can langsmith help with testing?"):
        print(f"Your input is: {input}")
        output_parser = StrOutputParser()

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are world class technical documentation writer."),
                ("user", "{input}"),
            ]
        )
        chain = prompt | self.llm | output_parser
        prompt = {"input": f"{input}"}
        print(chain.invoke(prompt))


if __name__ == "__main__":
    chat = OpenAIChain()
    chat.answer()
