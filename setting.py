import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_upstage import UpstageEmbeddings
from langchain_community.vectorstores import FAISS


def load_pdf():
    loader = PyMuPDFLoader("/home/aca123/project_1/food.pdf")
    doc = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=50)
    split_doc = splitter.split_documents(doc)

    return  split_doc


def embed(api_key):
    embeddings = UpstageEmbeddings(
        api_key=api_key,
        model="solar-embedding-1-large"
    )
    return embeddings

def search(split_doc, embeddings):
    vectorstore = FAISS.from_documents(documents=split_doc, embedding=embeddings)

    retriever = vectorstore.as_retriever(k=5) #검색기로 만들기
    return retriever