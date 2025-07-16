import os
import time
import uuid
import streamlit as st

from setting import load_pdf, embed,search
from dotenv import load_dotenv
from langchain_upstage import ChatUpstage
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

load_dotenv()
api_key = os.getenv("solar_key")

split_doc= load_pdf()
embeddings = embed(api_key)
retriever = search(split_doc, embeddings)

st.title("ᖗ( ᐛ )ᖘ")
st.caption("한식을 추천해 드려요!")

#세션 생성 및 초기화 세션을 실행할때마다 초기화되는데 입력한 정보들이 초기화되는걸 막아준다. 이전대화와 지금대화를 분리해서 저장시켜준다.
if "id" not in st.session_state:   
    st.session_state.id = uuid.uuid4()
    st.session_state.file_cache = {}

session_id = st.session_state.id
client = None

def reset_chat():
    st.session_state.messages = [] #이전에 나눈 대화를 저장해놓는 공간, 채팅 초기화 함수
    st.session_state.context = None

chat = ChatUpstage(upstage_api_key=api_key) #chat_bot 생성

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

qa_system_prompt = """당신은 한국의 전통 음식을 따뜻하고 친근하게 소개해주는 친절한 안내자 입니다. 당신의 목표는 사용자가 요청한 한국의 전통 음식을 알차게 설명하는 것입니다.
**반드시, 서론이나 인삿말을 한 후 첫 번째 음식 소개를 시작하고, 답변의 마무리에는 조언이나 인사말로 마무리 해주세요.**

사용자의 선호도나 기분은 다음과 같습니다:
{context}

다음 기준을 반드시 고려해 주세요:
- 추천할 한국 전통 음식은 **4~7가지**로 제한합니다. (사용자가 특정 지역/상황을 명시하지 않으면 보편적인 전통 음식을 추천합니다.)
- 각 음식의 **특징, 맛, 간단한 유래 또는 배경**을 설명하여 흥미를 유발합니다.
- 중복되는 음식이 있으면 하나만 설명해주세요.
- (선택 사항) 만약 해당 음식에 대한 사용자 기분 정보(context)가 있다면, 그에 맞춰 설명을 조절합니다.

**또한, 각 음식 설명 사이에는 반드시 구분선(---)을 넣어주세요.**
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

MAX_MESSAGES_BEFORE_DELETION = 8  #prompt비용처리

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

            with st.expander("참고한 자료"):
                st.write(result["context"])

            for chunk in result["answer"].split(" "):
                full_response += chunk + " "
                time.sleep(0.2)
                message_placeholder.markdown(full_response)


        st.session_state.messages.append(
            {"role": "assistant","content": full_response})
