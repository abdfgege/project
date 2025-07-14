import time
import os
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


st.title("ᖗ( ᐛ )ᖘ")


if "id" not in st.session_state:   #세션 생성 및 초기화 세션을 실행할때마다 초기화되는데 입력한 정보들이 초기화되는걸 막아준다. 이전대화와 지금대화를 분리해서 저장시켜준다.
    st.session_state.id = uuid.uuid4()
    st.session_state.file_cache = {}

session_id = st.session_state.id
client = None

def reset_chat():
    st.session_state.messages = [] #이전에 나눈 대화를 저장해놓는 공간, 채팅 초기화 함수
    st.session_state.context = None

loader = PyMuPDFLoader("/home/aca123/project_1/food.pdf")
doc = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=50)
split_doc = splitter.split_documents(doc)

embeddings = UpstageEmbeddings(
    api_key=api_key,
    model="solar-embedding-1-large"
)

vectorstore = FAISS.from_documents(documents=split_doc, embedding=embeddings)

retriever = vectorstore.as_retriever(k=4)  #검색기로 만들기


chat = ChatUpstage(upstage_api_key=os.getenv("solar_key")) #chat_bot 생성

contextualize_q_system_prompt = """사용자에게 받은 질문을 해석해서 지금 사용자가 어떤걸 선호하는지, 기분은 어떠한지를
차근차근 분석해 풀어써서 질문을 세세하게 답변하기 쉽게 재구성 시켜주세요."""

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

qa_system_prompt = """당신은 한국의 맛있는 음식을 추천해주는 따뜻하고 신뢰할 수 있는 AI 미식 비서입니다.

사용자의 선호도나 기분은 다음과 같습니다:
{context}

이 정보를 바탕으로 사용자에게 가장 적합한 한식, 특정 한식 요리 또는 한식 레스토랑을 추천해 주세요.

다음 기준을 반드시 고려해 주세요:
- 사용자의 감정 상태와 정서적 필요에 부합하는 분위기 (예: 아늑하고 편안함, 활기차고 신남, 위로가 되는 맛, 새로운 경험 등)
- 음식 섭취의 주된 목적 (예: 간단한 식사, 특별한 기념일, 스트레스 해소, 건강식, 미식 탐험, 현지 문화 체험 등)
- 현재 계절과 날씨 (예: 더운 날씨에는 시원한 한식, 추운 날씨에는 따뜻한 한식 등)
- 함께 식사하는 동행 여부 (혼자, 친구와, 연인과, 가족과 등) 및 이에 맞는 한식 또는 분위기 제안
- 예산과 시간의 제약이 있다면 그 범위 내에서 최적의 선택을 제공
- 특정 재료에 대한 선호나 알레르기 유무 (만약 정보가 있다면)

다음 사항을 지켜서 답변을 작성해 주세요:
- 정중하고 따뜻한 말투로, 마치 친한 친구처럼 자연스럽게 안내해 주세요
- 추천 한식, 요리는 4~5가지로 제한하고, 각각의 특징과 추천 이유를 구체적으로 설명해 주세요
- 이유를 어린아이들도 들으면 자세하게 이해할 수 있을정도록 풀어서 설명해 주세요
- 만약 적절한 한식 추천이 어렵다면, 솔직하게 "추천이 어렵습니다"라고 말해주세요
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

MAX_MESSAGES_BEFORE_DELETION = 10  #prompt비용처리

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


    st.session_state.messages.append(
        {"role": "assistant","content": full_response})
