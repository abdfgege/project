# 한식 추천 챗봇

## 프로젝트 소개

이 프로젝트는 RAG랑 Streamlit를 이용해서 
주제에 맞는 최선의 답을 꺼내올수있게 만든 챗봇입니다.

## 사용한 도구
- **Python** : 주요 개발 언어
- **Streamlit** : 웹 애플리케이션 UI구성
- **Upstage Embedding** : 문서 임베딩
- **LangChain** : LLM개발 프레임워크
- **FAISS** : 벡터 데이터베이스
- **PyMuPDFLoader** : PDF문서 로딩 및 처리
- **Dotenv** : 환경변수 관리

## 실행 방법

아래 지침에 따라 로컬 환경에서 챗봇을 설정하고 실행할 수 있습니다.

### 전제 조건

- Python 3.9+ 버전이 설치되어 있어야 합니다.
- Git이 설치되어 있어야 합니다.
- requirements.txt를 받아서 설치해야 합니다.
- solar_key는 직접 홈페이지에 들어가 받아서 .env 파일에 넣어줘야합니다.

### 실행시키기
```bash
git clone [https://github.com/abdfgege/project.git](https://github.com/abdfgege/project.git)
cd project/project_1 # 프로젝트 폴더 경로가 project_1이라면 이 명령어를 사용

solar_key="YOUR_UPSTAGE_SOLAR_API_KEY"(Solar홈페이지 들어가서 로그인후 키 받기)

streamlit run your_app_file_name.py(streamlit run Project.py)

```
## 간단 코드분석

### 코드내에 파일 불러오기
```bash
loader = PyMuPDFLoader("/home/aca123/project_1/food.pdf")
doc = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=50)
split_doc = splitter.split_documents(doc)

```


### 문서를 읽을 수 있게 임베딩 시키기
```bash
embeddings = UpstageEmbeddings(
    api_key=api_key,
    model="solar-embedding-1-large"
)

```


### 검색기로 만들기
```bash
vectorstore = FAISS.from_documents(documents=split_doc, embedding=embeddings)

retriever = vectorstore.as_retriever(k=5)

```


### 챗봇 및 프롬프트 생성
```bash
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

```


### 출력처리
```bash
if prompt := st.chat_input("채팅을 입력하세요 :)"):
    if len(st.session_state.messages) >= MAX_MESSAGES_BEFORE_DELETION:
        del st.session_state.messages[0]

```