import time
import os
import uuid
import tempfile
from langchain_upstage import UpstageEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader

from langchain_upstage import ChatUpstage
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
st.caption("한식을 추천해 드려요!")

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
splitter = RecursiveCharacterTextSplitter(chunk_size=1500,chunk_overlap=50)
split_doc = splitter.split_documents(doc)

embeddings = UpstageEmbeddings(
    api_key=api_key,
    model="solar-embedding-1-large"
)

vectorstore = FAISS.from_documents(documents=split_doc, embedding=embeddings)

retriever = vectorstore.as_retriever(k=5)  #검색기로 만들기


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

qa_system_prompt = """당신은 한국의 특정 지역에서 유명한 음식을 따뜻하고 친근하게 추천해주는 미식 비서입니다.

사용자의 선호도나 기분은 다음과 같습니다:
{context}

사용자가 특정 지역의 '유명한 음식'을 물어볼 경우, 해당 지역의 **대표적인 향토 음식 또는 특산물을 활용한 요리**를 추천해 주세요.

다음 기준을 반드시 고려해 주세요:
- 추천할 지역 특색 음식은 **3~7가지**로 제한합니다.
- 각 음식의 **특징, 맛, 간단한 유래 또는 배경**을 설명하여 흥미를 유발합니다.
- 사용자가 질문한 지역과 해당 음식이 **밀접한 관련**이 있어야 합니다.
- 중복되는 음식이 있으면 하나만 설명해주세요.
- (선택 사항) 만약 해당 지역에 대한 사용자 기분 정보(context)가 있다면, 그에 맞춰 음식을 선택하거나 설명을 조절합니다.

다음 사항을 지켜서 답변을 작성해 주세요:
- **친근하고 따뜻한 말투**로, 마치 현지 정보를 잘 아는 친구처럼 자연스럽게 안내해 주세요.
- 답변의 시작은 사용자의 방문/질문에 대한 **공감과 환영**으로 시작합니다.
- 추천하는 음식에 대한 **간결하고 매력적인 설명**을 제공합니다.
- 답변의 마무리는 사용자에게 도움이 되는 **간단한 조언이나 기분 좋은 인사**로 마무리합니다.
- 만약 적절한 지역 유명 음식 추천이 어렵다면, 솔직하게 "정보가 부족하여 추천이 어렵습니다"라고 말해주세요.
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
        with st.spinner("생각중...."):
            message_placeholder = st.empty()
            full_response = ""

            result = rag_chain.invoke({"input": prompt, "chat_history": st.session_state.messages})

            with st.expander("참고용"):
                st.write(result["context"])

            for chunk in result["answer"].split(" "):
                full_response += chunk + " "
                time.sleep(0.2)
                message_placeholder.markdown(full_response)


        st.session_state.messages.append(
            {"role": "assistant","content": full_response})
