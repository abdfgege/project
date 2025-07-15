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

## 실행방법
solar_key="YOUR_UPSTAGE_SOLAR_API_KEY"(Solar홈페이지 들어가서 로그인후 키 받기)

streamlit run your_app_file_name.py(streamlit run Project.py)