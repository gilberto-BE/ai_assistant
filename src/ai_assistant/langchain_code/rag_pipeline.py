from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough



class RAGPipeline:
    def __init__(self, llm, retriever, system_prompt=None):
        self.llm = llm  # Pass the ChatLLM instance here
        self.retriever = retriever
        self.system_prompt = (
            system_prompt or
            "You are an assistant for question-answering tasks. Use the following pieces of retrieved "
            "context to answer the question. If you don't know the answer, say that you don't know. "
            "Use three sentences maximum and keep the answer concise.\n\n{context}"
        )

    def format_documents(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def build_chain(self):
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                ("human", "{input}"),
            ]
        )
        return {
            "context": self.retriever | self.format_documents,
            "input": RunnablePassthrough(),
        } | prompt

    def query(self, user_input):
        rag_chain = self.build_chain()
        inputs = rag_chain.invoke({"input": user_input})
        response_stream = self.llm.stream_chat(inputs["context"] + "\n\n" + inputs["input"])

        # Stream tokens
        for token in response_stream:
            yield token
