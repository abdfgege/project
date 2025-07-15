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

## 🚀 설치 및 실행 방법

아래 지침에 따라 로컬 환경에서 챗봇을 설정하고 실행할 수 있습니다.

### 1. 전제 조건

- Python 3.9+ 버전이 설치되어 있어야 합니다.
- Git이 설치되어 있어야 합니다.

### 2. 프로젝트 클론

먼저, 이 저장소를 로컬 컴퓨터로 클론(Clone)합니다.

```bash
git clone [https://github.com/abdfgege/project.git](https://github.com/abdfgege/project.git)
cd project/project_1 # 프로젝트 폴더 경로가 project_1이라면 이 명령어를 사용

solar_key="YOUR_UPSTAGE_SOLAR_API_KEY"(Solar홈페이지 들어가서 로그인후 키 받기)

streamlit run your_app_file_name.py(streamlit run Project.py)