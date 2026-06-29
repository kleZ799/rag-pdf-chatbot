
import os

print("This project requires an OpenAI API key to run.")
print("Set it using: os.environ['OPENAI_API_KEY'] = 'your_key'")

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains.retrieval_qa.base import RetrievalQA

loader = PyPDFLoader("sample.pdf")
documents = loader.load()

splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(docs, embeddings)

retriever = db.as_retriever()

qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(),
    retriever=retriever
)

while True:
    query = input("Ask a question (or type 'exit'): ")
    if query.lower() == "exit":
        break
    print("Answer:", qa.run(query))