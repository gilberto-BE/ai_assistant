# import chunk
# from langchain_openai import ChatOpenAI
# from langchain_ollama import ChatOllama
# # from langchain_hf import ChatHF
# from langchain_core.messages import  SystemMessage, HumanMessage
# import bs4
# from langchain import hub
# from langchain_chroma import Chroma
# from langchain_community.document_loaders import WebBaseLoader
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough
# from langchain_openai import OpenAIEmbeddings
# from langchain_text_splitters import RecursiveCharacterTextSplitter


# # Define a system prompt that tells the model how to use the retrieved context
# system_prompt = """You are an assistant for question-answering tasks. 
# Use the following pieces of retrieved context to answer the question. 
# If you don't know the answer, just say that you don't know. 
# Use three sentences maximum and keep the answer concise.
# Context: {context}:"""

# def rag_model(question:str):
#     llm = ChatOpenAI(model='gpt-4o-mini')
#     return llm


# def get_loader():
#     loader = WebBaseLoader(web_paths="https://lilianweng.github.io/posts/2023-06-23-agent/", bs_kwargs=dict(parse_only=bs4.SoupStrainer(class_ =('post-content', 'post-title', 'post-header'))))
#     docs = loader.load()
#     return docs

# def split_text(docs):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024,chunk_overlap=200)
#     splits = text_splitter.split_documents(docs)
#     vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
#     retriever = vectorstore.as_retriever()
#     prompt = hub.pull('rml/rag-prompt')
#     return prompt, retriever
    
    
# def format_docs(docs):
#     return "\n\n".join(doc.page_content for doc in docs)


# if __name__ == "__main__":
#     rag_chain = ({'context': retriever | format_doct})


from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.llms import Ollama
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import WebBaseLoader


# Load documents for RAG. Here, we'll use a sample text file.
loader = WebBaseLoader('https://python.langchain.com/docs/concepts/rag/') # Replace with your document path
documents = loader.load()

# Split documents into smaller chunks to better store in vector database.
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(documents)

# Use OpenAI embeddings to convert text to vectors for similarity search
embeddings = OpenAIEmbeddings()  # You can use another embedding model here, e.g., local models if required

# Store document vectors in a FAISS vector store, which is a common choice for vector similarity
vectorstore = FAISS.from_documents(docs, embeddings)

# Initialize Ollama to use Llama 2. This requires that you have the Ollama setup configured correctly.
ollama_llm = Ollama(model='llama2')

# Create a RetrievalQA chain with Llama 2 via Ollama as the language model
rag_chain = RetrievalQA(llm=ollama_llm, retriever=vectorstore.as_retriever())

# Example query to ask a question from the documents
query = "What is the summary of the provided text?"
result = rag_chain.run(query)

print(result)

# To make this work, make sure:
# 1. Ollama is set up and running (ollama is the CLI tool to run models like Llama 2 locally).
# 2. You have Llama 2 or other Ollama models installed and ready to be served.
# 3. Replace 'sample_docs.txt' with the path to your text file.
