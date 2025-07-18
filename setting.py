import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from pathlib import Path
from langchain_upstage import UpstageEmbeddings
from langchain_community.vectorstores import FAISS


# 파일로더
def load_pdf():
    file_path = Path(__file__).parent / "food.pdf"
    if not file_path.exists():
        raise FileNotFoundError(f"{file_path} 파일이 존재하지 않습니다.")
    loader = PyMuPDFLoader(str(file_path))
    return loader.load()


def chunk():
    doc = load_pdf()

    # 로드된 문서를 청크로 분할
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_doc = splitter.split_documents(doc)
    
    return split_doc



# 임베딩
def embed(api_key):
    embeddings = UpstageEmbeddings(
        api_key=api_key,
        model="solar-embedding-1-large"
    )
    return embeddings


# 검색기
def search(split_doc, embeddings):
    vectorstore = FAISS.from_documents(documents=split_doc, embedding=embeddings)

    retriever = vectorstore.as_retriever(k=5) #검색기로 만들기
    return retriever