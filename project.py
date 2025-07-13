import time
import os
import base64
import uuid
import tempfile
from langchain_core.output_parsers import StrOutputParser
from langchain_upstage import UpstageEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader

from langchain_upstage import ChatUpstage
from langchain_core.messages import HumanMessage, SystemMessage

from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from openai import OpenAI 
from dotenv import load_dotenv
import streamlit as st


load_dotenv()
api_key = os.getenv("solar_key")


st.title("여행지를 추천해 드려요ᖗ( ᐛ )ᖘ")


if "id" not in st.session_state:   #세션 생성 및 초기화
    st.session_state.id = uuid.uuid4()
    st.session_state.file_cache = {}

    session_id = st.session_state.id
client = None

def reset_chat():
    st.session_state.messages = []
    st.session_state.context = None

loader = PyMuPDFLoader("/home/aca123/project_1/journey.pdf")
doc = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=5000,chunk_overlap=100)
split_doc = splitter.split_documents(doc)

embeddings = UpstageEmbeddings(
    api_key=api_key,
    model="solar-embedding-1-large"
)

vectorstore = FAISS.from_documents(documents=split_doc, embedding=embeddings)

retriever = vectorstore.as_retriever(k=8)  #검색기로 만들기


chat = ChatUpstage(upstage_api_key=os.getenv("solar_key")) #chat_bot 생성

contextualize_q_system_prompt = """사용자에게 받은 질문을 해석해서 지금 사용자의 감정은 어떠하고, 어떤 상황에 놓여있으며,
성격은 어떠한지, 그리고 현 상황을 분석해서 질문을 정제해주세요"""

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

history_aware_retriever = create_history_aware_retriever(
    chat, retriever, contextualize_q_prompt
)

qa_system_prompt = """당신은 사용자에게 여행지를 추천하는 친절한 AI 비서입니다.
사용자의 감정, 상황, 성격은 다음과 같습니다:
{context}

이 정보를 바탕으로 전세계들의 여행지 및 관광지 중에서 가장 적합한 여행지와 활동을 추천해 주세요.
만약 적절한 정보를 찾을 수 없다면 솔직히 모른다고 답하세요.
답변은 항상 한국어로 작성해주세요.
"""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(chat, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

MAX_MESSAGES_BEFORE_DELETION = 10

if prompt := st.chat_input("채팅을 입력하세요 :)"):
    if len(st.session_state.messages) >= MAX_MESSAGES_BEFORE_DELETION:
        del st.session_state.messages[0]

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        result = rag_chain.invoke({"input": prompt, "chat_history": st.session_state.messages})
        with st.expander("Evidence context"):
            st.write(result["context"])
        for chunk in result["answer"].split(" "):
            full_response += chunk + " "
            time.sleep(0.2)
            message_placeholder.markdown(full_response)