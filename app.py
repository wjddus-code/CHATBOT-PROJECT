import streamlit as st
import base64
import os
import requests
import re
import json
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# ============================================================
# 페이지 및 기본 설정
# ============================================================
st.set_page_config(
    page_title="인사담당자를 위한 법무 AI 챗봇",
    page_icon="🏛️",
    layout="wide",
)

# Document 폴더 자동 생성
if not os.path.exists("Document"):
    os.makedirs("Document")

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "search_history" not in st.session_state:
    st.session_state.search_history = []

# ============================================================
# 커스텀 CSS (All-White & Clean Blue 테마)
# ============================================================
st.markdown(
    """
<style>
    .stApp { background-color: #ffffff; }
    
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #f0f2f6;
    }

    .user-box {
        background-color: #0066cc; 
        color: white; 
        padding: 15px;
        border-radius: 20px 20px 5px 20px; 
        margin: 10px 0 10px 20%;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        font-size: 15px;
    }
    .ai-box {
        background-color: #f8f9fa; 
        color: #1a1a1a; 
        padding: 15px;
        border-radius: 20px 20px 20px 5px; 
        margin: 10px 20% 10px 0;
        border: 1px solid #e9ecef;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
        font-size: 15px;
    }

    .stButton>button {
        width: 100%;
        border-radius: 8px;
        border: 1px solid #0066cc;
        background-color: white;
        color: #0066cc;
        font-weight: 600;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #0066cc;
        color: white;
    }
    
    .stTextInput>div>div>input, .stTextArea>div>div>textarea {
        border-color: #e9ecef !important;
    }
    
    .search-result {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #0066cc;
    }
    .source-link {
        color: #0066cc;
        font-size: 0.9em;
    }
    
    .mode-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
        margin-bottom: 10px;
    }
    .mode-rag {
        background-color: #e8f5e9;
        color: #2e7d32;
    }
    .mode-web {
        background-color: #e3f2fd;
        color: #1565c0;
    }
    .mode-llm {
        background-color: #fff3e0;
        color: #e65100;
    }
</style>
""",
    unsafe_allow_html=True,
)


def get_base64_image(image_path):
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except:
        return None


# ============================================================
# RAG: 인덱싱 함수
# ============================================================
def perform_indexing():
    with st.spinner("Document 폴더 내 문서를 인덱싱 중입니다..."):
        try:
            loader = PyPDFDirectoryLoader("Document/")
            documents = loader.load()
            if not documents:
                st.warning("Document 폴더에 PDF 파일이 없습니다.")
                return
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=800, chunk_overlap=100
            )
            splits = text_splitter.split_documents(documents)
            embeddings = OpenAIEmbeddings(api_key=st.secrets["OPENAI_API_KEY"])
            vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
            st.session_state.vector_store = vectorstore
            st.success(f"인덱싱 완료! 총 {len(splits)}개의 지식 조각을 생성했습니다.")
        except Exception as e:
            st.error(f"인덱싱 중 오류 발생: {e}")


# ============================================================
# 웹 검색 함수
# ============================================================
def search_naver_blog(query: str, num_results: int = 10) -> list:
    """네이버 블로그 검색 API"""
    url = "https://openapi.naver.com/v1/search/blog.json"
    headers = {
        "X-Naver-Client-Id": st.secrets["NAVER_CLIENT_ID"],
        "X-Naver-Client-Secret": st.secrets["NAVER_CLIENT_SECRET"],
    }
    params = {
        "query": query,
        "display": num_results,
        "sort": "sim",
    }

    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        results = response.json()

        search_results = []
        for item in results.get("items", []):
            title = re.sub(r"<[^>]+>", "", item.get("title", ""))
            description = re.sub(r"<[^>]+>", "", item.get("description", ""))
            search_results.append(
                {
                    "title": title,
                    "link": item.get("link", ""),
                    "snippet": description,
                    "source": "네이버 블로그",
                    "date": item.get("postdate", ""),
                }
            )
        return search_results
    except Exception as e:
        return []


def search_naver_cafe(query: str, num_results: int = 10) -> list:
    """네이버 카페 검색 API"""
    url = "https://openapi.naver.com/v1/search/cafearticle.json"
    headers = {
        "X-Naver-Client-Id": st.secrets["NAVER_CLIENT_ID"],
        "X-Naver-Client-Secret": st.secrets["NAVER_CLIENT_SECRET"],
    }
    params = {"query": query, "display": num_results, "sort": "sim"}

    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        results = response.json()

        search_results = []
        for item in results.get("items", []):
            title = re.sub(r"<[^>]+>", "", item.get("title", ""))
            description = re.sub(r"<[^>]+>", "", item.get("description", ""))
            search_results.append(
                {
                    "title": title,
                    "link": item.get("link", ""),
                    "snippet": description,
                    "source": "네이버 카페",
                    "cafe_name": item.get("cafename", ""),
                }
            )
        return search_results
    except Exception as e:
        return []

def search_naver_news(query: str, num_results: int = 10) -> list:
    """네이버 뉴스 검색 API"""
    url = "https://openapi.naver.com/v1/search/newsarticle.json"
    headers = {
        "X-Naver-Client-Id": st.secrets["NAVER_CLIENT_ID"],
        "X-Naver-Client-Secret": st.secrets["NAVER_CLIENT_SECRET"],
    }
    params = {"query": query, "display": num_results, "sort": "sim"}

    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        results = response.json()

        search_results = []
        for item in results.get("items", []):
            title = re.sub(r"<[^>]+>", "", item.get("title", ""))
            description = re.sub(r"<[^>]+>", "", item.get("description", ""))
            search_results.append(
                {
                    "title": title,
                    "link": item.get("link", ""),
                    "snippet": description,
                    "source": "네이버 뉴스",
                    "news_name": item.get("newsname", ""),
                }
            )
        return search_results
    except Exception as e:
        return []


def search_web(query: str, sources: list, num_results: int = 5) -> list:
    """네이버 블로그 + 카페 + 뉴스 통합 검색"""
    all_results = []
    if "네이버 블로그" in sources:
        all_results.extend(search_naver_blog(query, num_results))
    if "네이버 카페" in sources:
        all_results.extend(search_naver_cafe(query, num_results))
    if "네이버 뉴스" in sources:
        all_results.extend(search_naver_news(query, num_results))
    return all_results


# ============================================================
# 질문 분류 함수
# ============================================================
def classify_query(query: str, has_vector_store: bool) -> str:
    """
    질문을 분류하여 RAG / LLM / 웹 검색으로 분기
    1. 주요 법무 관련 → RAG
    2. 그 외 → LLM이 판단 (AUTO)
    """
    # HR/법무 관련 키워드 (RAG 사용)
    rag_keywords = [
        "정규직", "계약직", "인턴", "근로", "근로계약서", "급여", "휴가", "휴직",
        "임금", "성과급", "상여금", "근태", "근로기준법", "휴게", "근로시간",
        "수습", "시용", "퇴사", "연말정산", "해고", "근태", "월급", "연봉", "주휴수당", "개정안", "정책"
    ]
    
    query_lower = query.lower()
    
    # RAG 키워드 체크
    for keyword in rag_keywords:
        if keyword in query_lower:
            return "RAG"
    
    # 그 외 질문은 LLM이 자동 판단하도록 AUTO 반환
    return "AUTO"


def determine_search_need(query: str, api_key: str) -> dict:
    """
    LLM을 사용하여 질문이 웹 검색이 필요한지 판단
    Returns: {"need_search": bool, "reason": str, "search_query": str}
    """
    llm = ChatOpenAI(
        model="gpt-5-mini",
        api_key=api_key,
        temperature=1,
    )
    
    classification_prompt = f"""당신은 질문 분류기입니다. 반드시 JSON 형식으로만 응답하세요.

[웹 검색이 필요한 질문 유형]
- 최신 뉴스, 현재 시세, 실시간 정보
- 특정 장소/상품/서비스 후기, 리뷰
- 날씨, 주가, 환율 등 실시간 데이터
- 특정 기업/인물의 최근 소식
- 부동산/아파트 정보 (임장 후기, 시세, 분양)
- 최근 이벤트, 행사 정보

[웹 검색이 필요 없는 질문 유형]
- 일반 지식, 개념 설명
- 코딩, 프로그래밍 도움
- 수학, 과학 등 보편적 지식
- 번역, 문법 교정
- 창작, 글쓰기
- 일반적인 조언

질문: "{query}"

위 질문을 분석하여 아래 JSON 형식으로만 응답하세요. 다른 텍스트는 절대 포함하지 마세요:
{{"need_search": true, "reason": "이유", "search_query": "검색어"}}
또는
{{"need_search": false, "reason": "이유", "search_query": ""}}"""
    
    try:
        response = llm.invoke([HumanMessage(content=classification_prompt)])
        result_text = response.content.strip()
        
        # ```json 등의 마크다운 제거
        if "```" in result_text:
            result_text = re.sub(r'```json\s*', '', result_text)
            result_text = re.sub(r'```\s*', '', result_text)
            result_text = result_text.strip()
        
        # JSON 파싱 시도
        result = json.loads(result_text)
        
        # 필수 키 검증
        if "need_search" not in result:
            result["need_search"] = False
        if "reason" not in result:
            result["reason"] = "자동 판단"
        if "search_query" not in result:
            result["search_query"] = ""
            
        return result
    except json.JSONDecodeError:
        # JSON 파싱 실패 시 텍스트에서 판단 시도
        result_lower = response.content.lower() if response else ""
        if "true" in result_lower or "필요" in result_lower:
            return {"need_search": True, "reason": "웹 검색 필요로 판단", "search_query": query}
        return {"need_search": False, "reason": "AI 직접 답변 가능", "search_query": ""}
    except Exception as e:
        # 기타 오류 시 기본값 반환
        return {"need_search": False, "reason": f"판단 중 오류: {str(e)}", "search_query": ""}


# ============================================================
# 대표 질문용 미리 정의된 답변
# ============================================================
PREDEFINED_ANSWERS = {
    "🏛️ 2026년 최저임금을 알려줘.": """

* **시급**: 10,320원(2025 대비 2.9% 인상)
* **월급**: 2,156,880원(주 40시간 기준, 주휴 수당 포함)
* **대상**: 1인 이상 근로자를 사용하는 모든 사업 또는 사업장. 근로기준법상 근로자. 정규직, 비정규직, 파트타임, 아르바이트, 청소년 근로자, 외국인 근로자 등
    """,
    "📅 근로기준법상 휴게시간을 알려줘.": """
4시간 근무 시 30분 이상, 8시간 근무 시 1시간 이상을 근로시간 도중에 부여해야 합니다.정규직, 비정규직, 파트타임, 아르바이트, 청소년 근로자, 외국인 근로자 등

1. **주요 기준**: 휴게시간은 근로시간에서 제외되며, 1일 법정 근로시간은 휴게시간을 제외하고 8시간, 1주 법정 근로시간은 휴게시간을 제외하고 40시간입니다.
2. **휴게 시간 단위**: 반드시 30분/1시간 단위로 부여해야 하는 것은 아닙니다. 다만, 너무 짧게(예 : 10분 미만) 나누어 주는 것은 근로자의 생존권을 침해하는 것으로 보아 허용되지 않으니 주의하셔야 합니다.
3. **예외 및 주의사항**: 5인 미만 사업장에도 근로기준법이 단계적으로 확대 적용될 수 있으므로, 사업장 규모에 따라 적용 여부가 달라질 수 있습니다.
휴게시간을 출근 전이나 퇴근 후에 몰아서 제공하는 것은 위법이며, 반드시 근로시간 중간에 배정해야 합니다.

**상세 일정**은 사이드바에서 [문서 인덱싱]을 완료하신 후, 질문해 주시면 학습된 가이드북을 토대로 더 자세히 안내해 드립니다!
    """,
    "💻 2026년 법정공휴일은 총 몇일인가요?": """
주말과 겹치지 않는 2026년 공휴일은 총 14일로, 상세 일정은 아래와 같습니다.

* **1월**: 1월 1일(목) 신정
* **2월**: 2월 16.17.18일(월.화.수) 설날 연휴
* **3월**: 3월 2일(월) 삼일절 대체공휴일
* **5월**: 5월 5일(화) 어린이날, 5월 25일(월) 부처님오신날 대체공휴일
* **6월**: 6월 3일(수) 제9회 전국동시지방선거
* **8월**: 8월 17일(월) 광복절 대체공휴일
* **9월**: 9월 24.25.26일(목.금.토) 추석 연휴
* **10월**: 10월 5일(월) 개천절 대체공휴일, 10월 9일(금) 한글날
* **12월**: 12월 25일(금) 크리스마스
    """,
}

# ============================================================
# 사이드바
# ============================================================
with st.sidebar:
    logo_b64 = get_base64_image("chatbot_logo.png")
    if logo_b64:
        st.markdown(
            f'<img src="data:image/png;base64,{logo_b64}" width="100%">',
            unsafe_allow_html=True,
        )
    else:
        st.title("🏛️ SeSAC AI")

    st.divider()
    
    # 지식 데이터베이스 섹션
    st.subheader("📚 지식 데이터베이스")
    if st.button("문서 인덱싱 시작"):
        perform_indexing()
    if st.session_state.vector_store:
        st.caption("✅ 문서 학습 완료")

    st.divider()
    
    # 웹 검색 설정 섹션
    st.subheader("🔍 웹 검색 설정")
    search_sources = st.multiselect(
        "검색 소스",
        ["네이버 블로그", "네이버 카페", "네이버 뉴스"],
        default=["네이버 블로그", "네이버 카페", "네이버 뉴스"],
    )
    num_results = st.slider("소스별 검색 결과 수", 3, 15, 5)
    
    st.divider()
    
    # AI 페르소나 설정
    st.subheader("AI 페르소나 설정")
    system_instruction = st.text_area(
        "AI 역할 정의:",
        value="너는 기업의 인사팀을 위한 전문 상담 AI야. 제공된 [Context]를 참고하여 인사담당자들에게 친절하고 정확하게 답변해줘. 웹 검색 결과가 제공되면 해당 정보를 바탕으로 종합적으로 분석해줘.",
        height=150,
    )
    
    st.divider()
    
    if st.button("대화 초기화"):
        st.session_state.messages = []
        st.session_state.search_history = []
        st.rerun()
    
    # 통계 표시
    st.divider()
    st.subheader("📊 사용 통계")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("대화 수", len(st.session_state.messages) // 2)
    with col2:
        st.metric("웹 검색", len(st.session_state.search_history))

# ============================================================
# 메인 화면
# ============================================================
st.markdown(
    "<h2 style='color: #0066cc;'>인사담당자를 위한 법무 AI 챗봇</h2>", unsafe_allow_html=True
)
st.caption("🚀 인사/법무 AI 챗봇 | 여러분의 검색 시간을 줄여드립니다")

st.markdown("### 자주 묻는 질문")
col1, col2, col3 = st.columns(3)
q1 = "🏛️ 2026년 최저임금을 알려줘."
q2 = "📅 근로기준법상 휴게시간을 알려줘."
q3 = "💻 2026년 법정공휴일은 총 몇일인가요?"

clicked_q = None
if col1.button("📍 2026년 최저임금"):
    clicked_q = q1
if col2.button("📋 근로기준법상 휴게시간"):
    clicked_q = q2
if col3.button("🙋 2026년 법정공휴일"):
    clicked_q = q3

st.divider()

# 대화 기록 표시
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        st.markdown(
            f'<div class="user-box">{msg.content}</div>', unsafe_allow_html=True
        )
    elif isinstance(msg, AIMessage):
        st.markdown(f'<div class="ai-box">{msg.content}</div>', unsafe_allow_html=True)

user_input = st.chat_input("질문을 입력해주세요. (예: 정규직과 계약직 차이, 통상임금 검색)")
final_query = clicked_q if clicked_q else user_input

if final_query:
    st.markdown(f'<div class="user-box">{final_query}</div>', unsafe_allow_html=True)
    st.session_state.messages.append(HumanMessage(content=final_query))

    # 답변 생성 로직
    if final_query in PREDEFINED_ANSWERS:
        # 미리 정의된 답변
        ai_content = PREDEFINED_ANSWERS[final_query]
        mode_badge = '<span class="mode-badge mode-rag">📚 사전 정의 답변</span>'
    else:
        # 질문 분류
        query_type = classify_query(final_query, st.session_state.vector_store is not None)
        
        try:
            if query_type == "RAG":
                # RAG 모드 (SeSAC/교육 관련)
                mode_badge = '<span class="mode-badge mode-rag">📚 RAG 모드 (교육 정보)</span>'
                
                context = ""
                if st.session_state.vector_store:
                    docs = st.session_state.vector_store.similarity_search(final_query, k=3)
                    context = "\n\n".join([doc.page_content for doc in docs])

                llm = ChatOpenAI(
                    model="gpt-5-mini",
                    api_key=st.secrets["OPENAI_API_KEY"],
                    streaming=True,
                    temperature=1,
                )

                full_system_prompt = f"{system_instruction}\n\n[Context]\n{context if context else '관련 문서 없음'}"
                prompt = [
                    SystemMessage(content=full_system_prompt)
                ] + st.session_state.messages

                with st.spinner("답변 생성 중..."):
                    response = llm.invoke(prompt)
                    ai_content = response.content
                    
            else:
                # AUTO 모드: LLM이 웹 검색 필요 여부 판단
                with st.spinner("질문 분석 중..."):
                    search_decision = determine_search_need(final_query, st.secrets["OPENAI_API_KEY"])
                
                if search_decision["need_search"]:
                    # 웹 검색 모드
                    mode_badge = '<span class="mode-badge mode-web">🔍 웹 검색 모드</span>'
                    
                    search_query = search_decision["search_query"] if search_decision["search_query"] else final_query
                    
                    with st.status(f"🔍 웹에서 '{search_query}' 검색 중...", expanded=True) as status:
                        all_results = []
                        seen_links = set()
                        
                        # 검색 실행
                        results = search_web(search_query, search_sources, num_results)
                        
                        for result in results:
                            if result["link"] not in seen_links:
                                seen_links.add(result["link"])
                                all_results.append(result)
                        
                        st.write(f"✅ {len(all_results)}개의 결과를 찾았습니다.")
                        st.caption(f"💡 판단 이유: {search_decision['reason']}")
                        status.update(label="검색 완료!", state="complete")
                    
                    # 검색 결과 표시
                    if all_results:
                        with st.expander("📑 검색된 원본 자료 보기", expanded=False):
                            for i, result in enumerate(all_results[:10], 1):
                                st.markdown(
                                    f"""
                                <div class="search-result">
                                    <strong>{i}. {result['title']}</strong><br>
                                    <span class="source-link">🔗 <a href="{result['link']}" target="_blank">{result['source']}</a></span><br>
                                    <small>{result['snippet'][:200]}...</small>
                                </div>
                                """,
                                    unsafe_allow_html=True,
                                )
                        
                        # 검색 기록 저장
                        st.session_state.search_history.append({
                            "query": search_query,
                            "results_count": len(all_results),
                        })
                    
                    # 웹 검색 결과를 컨텍스트로 구성
                    web_context = ""
                    for i, result in enumerate(all_results, 1):
                        web_context += f"\n[결과 {i}]\n"
                        web_context += f"제목: {result['title']}\n"
                        web_context += f"출처: {result['source']}\n"
                        web_context += f"링크: {result['link']}\n"
                        web_context += f"내용: {result['snippet']}\n"
                    
                    # LLM으로 웹 검색 결과 분석
                    llm = ChatOpenAI(
                        model="gpt-5-mini",
                        api_key=st.secrets["OPENAI_API_KEY"],
                        streaming=True,
                        temperature=1,
                    )
                    
                    web_system_prompt = f"""{system_instruction}

아래는 사용자 질문과 관련된 웹 검색 결과입니다. 이 정보를 바탕으로 종합적으로 분석하여 답변해주세요.
답변 시 출처 링크를 함께 표시해주세요.

[웹 검색 결과]
{web_context if web_context else '검색 결과 없음'}"""

                    prompt = [
                        SystemMessage(content=web_system_prompt)
                    ] + st.session_state.messages
                    
                    with st.spinner("답변 생성 중..."):
                        response = llm.invoke(prompt)
                        ai_content = response.content
                else:
                    # 일반 LLM 모드 (웹 검색 불필요)
                    mode_badge = '<span class="mode-badge" style="background-color:#fff3e0;color:#e65100;">🧠 AI 직접 답변</span>'
                    
                    llm = ChatOpenAI(
                        model="gpt-5-mini",
                        api_key=st.secrets["OPENAI_API_KEY"],
                        streaming=True,
                        temperature=1,
                    )
                    
                    # 일반 답변용 시스템 프롬프트 (웹 검색 언급 제거)
                    general_system_prompt = "너는 친절하고 유능한 AI 어시스턴트야. 사용자의 질문에 정확하고 도움이 되는 답변을 제공해줘."

                    prompt = [
                        SystemMessage(content=general_system_prompt)
                    ] + st.session_state.messages

                    with st.spinner("답변 생성 중..."):
                        response = llm.invoke(prompt)
                        ai_content = response.content
                    
        except Exception as e:
            ai_content = f"오류가 발생했습니다: {e}"
            mode_badge = '<span class="mode-badge" style="background-color:#ffebee;color:#c62828;">⚠️ 오류</span>'

    # 답변 표시
    st.markdown(mode_badge, unsafe_allow_html=True)
    st.markdown(f'<div class="ai-box">{ai_content}</div>', unsafe_allow_html=True)
    st.session_state.messages.append(AIMessage(content=ai_content))

# 하단 안내
st.divider()
st.caption(
    """
💡 **사용 안내**: 
- **일반 인사/법무 지식 질문**: AI 직접 답변
- **최신 정보 필요**: 뉴스, 블로그 리뷰, 최신 자료 등 → 🔍 웹 검색 모드 (AI가 자동 판단)
- 사이드바에서 [문서 인덱싱]을 완료하면 더 정확한 인사법무 관련 답변을 받을 수 있습니다.
"""
)