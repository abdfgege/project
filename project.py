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
splitter = RecursiveCharacterTextSplitter(chunk_size=3000,chunk_overlap=50)
split_doc = splitter.split_documents(doc)

embeddings = UpstageEmbeddings(
    api_key=api_key,
    model="solar-embedding-1-large"
)

vectorstore = FAISS.from_documents(documents=split_doc, embedding=embeddings)

retriever = vectorstore.as_retriever(k=10)  #검색기로 만들기


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

qa_system_prompt = """당신은 여행지를 추천해주는 따뜻하고 신뢰할 수 있는 AI 비서입니다.

사용자의 감정 상태, 현재 상황, 성격은 다음과 같습니다:
{context}

이 정보를 바탕으로 전 세계 여행지 중 사용자에게 가장 적합한 여행지와 활동을 추천해 주세요.

다음 기준을 반드시 고려해 주세요:
- 사용자의 감정 상태와 정서적 필요에 부합하는 분위기 (예: 고요함, 활력, 위로, 설렘 등)
- 여행의 주된 목적 (예: 휴식, 재충전, 치유, 모험, 문화 체험, 자연 속 고요함 등)
- 계절과 현지 기후 조건 (예: 너무 덥거나 추운 지역은 피하고, 계절에 어울리는 여행지 제안)
- 여행 동행 여부 (혼자, 친구와, 연인과, 가족과 등) 및 이에 맞는 활동 제안
- 예산과 일정의 제약이 있다면 그 범위 내에서 최적의 선택을 제공

다음 사항을 지켜서 답변을 작성해 주세요:
- 정중하고 따뜻한 말투로, 마치 친한 친구처럼 자연스럽게 안내해 주세요
- 추천 여행지는 1~2곳으로 제한하고, 각각의 분위기와 활동을 구체적으로 설명해 주세요
- 만약 적절한 여행지를 판단하기 어렵다면, 솔직하게 "추천이 어렵습니다"라고 말해주세요
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

MAX_MESSAGES_BEFORE_DELETION = 8

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
    