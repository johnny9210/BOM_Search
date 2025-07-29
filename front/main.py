import streamlit as st
import os
import sys
import tempfile
import json
from dotenv import load_dotenv

# .env 파일 로드 (프로젝트 루트에서)
load_dotenv()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from file.upstage import process_document_with_upstage
from file.search import create_opensearch_client, save_chunks_to_opensearch
from check.check_data import tech_sections, QA_sections
from rag.rag import rag_query, generate_answer_with_llm, create_llm_client



ip_address = st.context.ip_address

if ip_address:
    st.write(f"Client IP Address: {ip_address}")
else:
    st.write("No IP address found. This is expected during local development.")

    
def document_processing_page():
    """문서 디지털화 시스템 페이지"""
    st.title("📄 문서 디지털화 시스템")
    st.markdown("---")
    
    # 사이드바에서 API 키 설정
    with st.sidebar:
        st.header("⚙️ 설정")
        
        # 환경변수에서 API 키 확인
        env_api_key = os.getenv("UPSTAGE_API_KEY")
        
        if env_api_key and env_api_key != "UPSTAGE_API_KEY":
            api_key = env_api_key
                
        else:
            st.warning("⚠️ 환경변수에 API 키가 없습니다.")
            st.info("프로젝트 루트에 .env 파일을 생성하고 UPSTAGE_API_KEY=your_key_here를 추가하세요.")
            
            api_key = st.text_input(
                "Upstage API Key", 
                type="password",
                help="환경변수가 없으면 여기에 직접 입력하세요."
            )
            
            if api_key:
                st.success("✅ API 키가 입력되었습니다!")
        
    # 메인 컨텐츠
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 파일 업로드")
        
        uploaded_file = st.file_uploader(
            "처리할 문서를 업로드하세요",
            type=['pdf', 'png', 'jpg', 'jpeg', 'tiff', 'bmp'],
            help="지원 형식: PDF, PNG, JPG, JPEG, TIFF, BMP"
        )
        
        if uploaded_file is not None:
            # 파일 정보 표시
            
            
            
            # 처리 버튼
            if st.button("🚀 문서 처리 시작", type="primary", use_container_width=True):
                # 처리 결과를 session_state에 저장
                result = process_document(uploaded_file, api_key)
                if result:
                    st.session_state.processing_result = result
                    st.session_state.uploaded_file_name = uploaded_file.name
                    st.session_state.uploaded_file_size = uploaded_file.size
    
    # 처리 결과가 session_state에 있으면 표시
    if 'processing_result' in st.session_state:
        display_results(st.session_state.processing_result, col2)

def checklist_page():
    """체크리스트 페이지"""
    st.title("📝 Checklist")
    st.markdown("---")
    
    # 체크리스트 유형 선택
    checklist_type = st.radio(
        "체크리스트 유형 선택:",
        ["📋 기술 사양 체크리스트", "🔍 품질보증 체크리스트"],
        horizontal=True
    )
    
    st.markdown("---")
    
    if checklist_type == "📋 기술 사양 체크리스트":
        display_tech_checklist()
    else:
        display_qa_checklist()

def display_tech_checklist():
    """기술 사양 체크리스트 표시"""
    st.subheader("🔧 기술 사양 체크리스트")
    st.info("각 섹션의 요구사항이 문서에 포함되어 있는지 확인하세요.")
    
    # 각 기술 섹션별로 체크리스트 생성
    for section_name, items in tech_sections.items():
        with st.expander(f"📑 {section_name}", expanded=False):
            section_checks = []
            for i, item in enumerate(items):
                check_key = f"tech_{section_name}_{i}"
                checked = st.checkbox(item, key=check_key)
                section_checks.append(checked)
            
            # 섹션별 진행률
            section_progress = sum(section_checks) / len(section_checks) if section_checks else 0
            st.progress(section_progress)
            st.caption(f"섹션 완료율: {section_progress*100:.0f}% ({sum(section_checks)}/{len(section_checks)})")
    
def display_qa_checklist():
    """품질보증 체크리스트 표시"""
    st.subheader("🔍 품질보증 체크리스트")
    st.info("품질보증 요구사항이 문서에 포함되어 있는지 확인하세요.")
    
    # 각 QA 섹션별로 체크리스트 생성
    for section_name, items in QA_sections.items():
        with st.expander(f"📋 {section_name}", expanded=False):
            section_checks = []
            for i, item in enumerate(items):
                check_key = f"qa_{section_name}_{i}"
                checked = st.checkbox(item, key=check_key)
                section_checks.append(checked)
            
            # 섹션별 진행률
            section_progress = sum(section_checks) / len(section_checks) if section_checks else 0
            st.progress(section_progress)
            st.caption(f"섹션 완료율: {section_progress*100:.0f}% ({sum(section_checks)}/{len(section_checks)})")

def bom_qa_page():
    """BOM QA 채팅 페이지"""
    st.title("🤖 BOM QA")
    st.markdown("---")
    st.info("📚 문서에 대해 질문하고 AI가 답변해 드립니다.")
    
    # OpenSearch 연결 상태 확인
    try:
        opensearch_client = create_opensearch_client()
        llm_client = create_llm_client()
        
        if not llm_client:
            st.error("❌ LLM 클라이언트 연결에 실패했습니다. 환경변수를 확인해주세요.")
            return
            
        st.success("✅ OpenSearch 및 LLM 연결 완료")
    except Exception as e:
        st.error(f"❌ 서비스 연결 실패: {str(e)}")
        return
    
    # 채팅 히스토리 초기화
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # 채팅 히스토리 표시
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # 사용자 입력
    if prompt := st.chat_input("문서에 대해 질문해주세요..."):
        # 사용자 메시지 추가 및 표시
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # AI 응답 생성
        with st.chat_message("assistant"):
            with st.spinner("답변을 생성중입니다..."):
                try:
                    # RAG 검색 수행
                    result = rag_query(
                        question=prompt,
                        client=opensearch_client,
                        search_type="hybrid",
                        context_size=5
                    )
                    
                    if result['context']:
                        # LLM으로 답변 생성
                        answer = generate_answer_with_llm(
                            question=prompt,
                            context=result['context'],
                            llm_client=llm_client
                        )
                        
                        # 답변 표시
                        st.markdown(answer)
                        
                        # 참고 정보 표시
                        with st.expander("📊 검색 정보"):
                            chunk_ids = [str(res.get('chunk_id', 'unknown')) for res in result['search_results'] if res.get('content')]
                            st.write(f"**참고한 청크:** {', '.join(chunk_ids[:5])}")
                            st.write(f"**총 검색 결과 수:** {result['search_metadata']['total_results']}")
                            
                            # 검색된 청크 내용 미리보기
                            st.write("**검색된 청크 미리보기:**")
                            for i, res in enumerate(result['search_results'][:3], 1):
                                content = res.get('content', '')
                                if content:
                                    preview = content[:200] + "..." if len(content) > 200 else content
                                    st.text_area(f"청크 {res.get('chunk_id', 'unknown')}", preview, height=100, key=f"preview_{i}")
                        
                        # 세션에 AI 응답 저장
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                        
                    else:
                        error_msg = "관련된 정보를 찾을 수 없습니다. 다른 질문을 시도해보세요."
                        st.markdown(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
                        
                except Exception as e:
                    error_msg = f"답변 생성 중 오류가 발생했습니다: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
    
    # 채팅 초기화 버튼
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🗑️ 채팅 초기화", type="secondary", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    
    # 사이드바에 사용 팁 추가
    with st.sidebar:
        st.markdown("---")
        st.header("💡 사용 팁")
        st.markdown("""
        **효과적인 질문 방법:**
        - 구체적인 키워드 사용
        - 섹션 번호나 제목 언급
        - 기술 용어나 규격 번호 포함
        
        **예시 질문:**
        - "FORGING에 대해 알려줘"
        - "7.3.5 조항 내용"
        - "압력 용기 검사 요구사항"
        - "품질보증 절차"
        """)

def main():
    st.set_page_config(
        page_title="문서 처리 시스템",
        page_icon="📄",
        layout="wide"
    )
    
    # 사이드바에서 페이지 선택
    with st.sidebar:
        st.title("📄 Navigation")
        page = st.selectbox(
            "페이지 선택",
            ["📄 문서 디지털화 시스템", "📝 Checklist", "🤖 BOM QA"],
            key="page_selector"
        )
    
    # 선택된 페이지 실행
    if page == "📄 문서 디지털화 시스템":
        document_processing_page()
    elif page == "📝 Checklist":
        checklist_page()
    elif page == "🤖 BOM QA":
        bom_qa_page()
    
    
def process_document(uploaded_file, api_key):
    """업로드된 파일을 처리하는 함수"""
    
    # 진행 상태 표시
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("📁 임시 파일 생성 중...")
        progress_bar.progress(20)
        
        # 임시 파일로 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        status_text.text("🔄 Upstage API 호출 중...")
        
        # API 키 확인
        env_api_key = os.getenv("UPSTAGE_API_KEY")
        final_api_key = api_key or env_api_key
        
        if not final_api_key or final_api_key == "UPSTAGE_API_KEY":
            st.error("❌ API 키가 설정되지 않았습니다.")
            st.info("프로젝트 루트에 .env 파일을 생성하거나 사이드바에서 API 키를 입력해주세요.")
            return None
        
        # 문서 처리 (기본 설정 사용)
        result = process_document_with_upstage(
            file_path=tmp_file_path,
            api_key=final_api_key
        )
        
        status_text.text("✅ 처리 완료!")
        progress_bar.progress(100)
        
        return result
        
    except Exception as e:
        status_text.text("❌ 처리 중 오류 발생")
        progress_bar.progress(0)
        st.error(f"오류가 발생했습니다: {str(e)}")
        return None
        
    finally:
        # 임시 파일 정리
        try:
            if 'tmp_file_path' in locals():
                os.unlink(tmp_file_path)
        except:
            pass

def display_results(result, result_col):
    """처리 결과를 표시하는 함수"""
    
    with result_col:
        st.success("🎉 문서 처리가 성공적으로 완료되었습니다!")
        
        # 결과를 탭으로 구분하여 표시
        tab1, tab2, tab3, tab4 = st.tabs(["📋 요약", "📑 청크 분할", "🔍 상세 결과", "💾 다운로드"])
        
        with tab1:
            st.subheader("처리 요약")
            
            # 처리 통계 정보
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("청크 수", result.get('chunks_count', 0))
            with col2:
                st.metric("임베딩 수", result.get('embeddings_count', 0))
            with col3:
                if 'detected_headers' in result:
                    st.metric("감지된 헤더", len(result['detected_headers']))
                else:
                    st.metric("감지된 헤더", 0)
            with col4:
                st.metric("처리 상태", "✅ 완료" if result.get('chunks_count', 0) > 0 else "⚠️ 미완료")
            
            # 원본 텍스트 표시
            if 'content' in result and result['content']:
                # content가 딕셔너리인 경우와 문자열인 경우 처리
                if isinstance(result['content'], dict):
                    content_text = result['content'].get('text', '') or result['content'].get('html', '')
                else:
                    content_text = str(result['content'])
                    
                st.text_area("추출된 텍스트", content_text, height=200)
            else:
                st.info("추출된 텍스트가 없습니다.")
                
            # 임베딩 정보 표시
            if 'embeddings' in result and result['embeddings']:
                st.subheader("🔗 임베딩 정보")
                st.info(f"✅ {len(result['embeddings'])}개 청크에 대한 임베딩 벡터가 생성되었습니다.")
                if result['embeddings']:
                    st.text(f"벡터 차원: {len(result['embeddings'][0])}차원")
            elif 'embeddings_error' in result:
                st.subheader("🔗 임베딩 정보")
                st.error(f"❌ 임베딩 생성 실패: {result['embeddings_error']}")
            else:
                st.subheader("🔗 임베딩 정보")
                st.warning("⚠️ 임베딩이 생성되지 않았습니다.")
        
        with tab2:
            st.subheader("📑 청크 분할 결과")
            if 'chunks' in result and result['chunks']:
                st.success(f"✅ 총 {result.get('chunks_count', len(result['chunks']))}개의 청크로 분할되었습니다!")
                
                # OpenSearch 저장 버튼 추가 - session_state를 사용하여 저장 상태 관리
                col_a, col_b = st.columns([2, 1])
                with col_a:
                    st.info("💾 OpenSearch에 청크를 저장할 수 있습니다.")
                with col_b:
                    if st.button("🔄 OpenSearch에 저장", type="secondary", key="save_to_opensearch"):
                        try:
                            # OpenSearch 클라이언트 생성
                            opensearch_client = create_opensearch_client()
                            
                            # 메타데이터 준비 - session_state에서 파일 정보 가져오기
                            metadata = {
                                "file_name": st.session_state.get('uploaded_file_name', 'unknown'),
                                "file_size": st.session_state.get('uploaded_file_size', 0),
                                "chunks_count": len(result['chunks'])
                            }
                            
                            # 청크 저장 (임베딩 포함)
                            with st.spinner("OpenSearch에 저장 중..."):
                                embeddings = result.get('embeddings', [])
                                saved_ids = save_chunks_to_opensearch(
                                    chunks=result['chunks'],
                                    client=opensearch_client,
                                    document_name=st.session_state.get('uploaded_file_name', 'unknown'),
                                    metadata=metadata,
                                    embeddings=embeddings
                                )
                            
                            st.success(f"✅ {len(saved_ids)}개 청크가 OpenSearch에 저장되었습니다!")
                            if embeddings:
                                st.info(f"🔗 {len(embeddings)}개 임베딩 벡터도 함께 저장되었습니다.")
                            st.text(f"저장된 문서 ID 예시: {saved_ids[0] if saved_ids else 'N/A'}")
                            # 저장 상태를 session_state에 기록
                            st.session_state.save_success = True
                            
                        except Exception as e:
                            st.error(f"❌ OpenSearch 저장 실패: {str(e)}")
                
                # 저장 성공 메시지 표시
                if st.session_state.get('save_success', False):
                    st.info("💾 이 문서의 청크들이 이미 OpenSearch에 저장되었습니다.")
                
                # 감지된 헤더 표시 (디버깅)
                if 'detected_headers' in result:
                    with st.expander("🔍 감지된 헤더 목록"):
                        st.write("문서에서 자동으로 감지된 섹션 헤더들:")
                        for i, header in enumerate(result['detected_headers'], 1):
                            st.text(f"{i}. {header}")
                
                # 청크 선택기 - key 추가하여 상태 유지
                chunk_idx = st.selectbox(
                    "청크 선택:",
                    range(len(result['chunks'])),
                    format_func=lambda x: f"청크 {x+1}",
                    key="chunk_selector"
                )
                
                # 선택된 청크 표시
                if result['chunks']:
                    st.text_area(
                        f"청크 {chunk_idx + 1} 내용:",
                        result['chunks'][chunk_idx],
                        height=300,
                        key=f"chunk_content_{chunk_idx}"
                    )
                
                # 모든 청크 미리보기
                with st.expander("모든 청크 미리보기"):
                    for i, chunk in enumerate(result['chunks'], 1):
                        st.markdown(f"**--- 청크 {i} ---**")
                        st.text(chunk[:200] + "..." if len(chunk) > 200 else chunk)
                        st.markdown("---")
                        
            elif 'chunks_error' in result:
                st.error(f"❌ 청크 분할 오류: {result['chunks_error']}")
            else:
                st.info("청크로 분할할 HTML 컨텐츠가 없습니다.")
        
        with tab3:
            st.subheader("전체 응답 데이터")
            st.json(result)
        
        with tab4:
            st.subheader("결과 다운로드")
            
            # JSON 파일로 다운로드
            json_str = json.dumps(result, ensure_ascii=False, indent=2)
            st.download_button(
                label="📥 JSON 파일 다운로드",
                data=json_str,
                file_name=f"processed_{st.session_state.get('uploaded_file_name', 'unknown')}.json",
                mime="application/json"
            )
            
            # 청크 텍스트 파일로 다운로드
            if 'chunks' in result and result['chunks']:
                chunks_text = "\n\n".join([f"--- 청크 {i} ---\n{chunk}" for i, chunk in enumerate(result['chunks'], 1)])
                st.download_button(
                    label="📥 청크 텍스트 파일 다운로드",
                    data=chunks_text,
                    file_name=f"chunks_{st.session_state.get('uploaded_file_name', 'unknown')}.txt",
                    mime="text/plain"
                )
            
            # 원본 텍스트 파일로 다운로드 (content가 있는 경우)
            if 'content' in result and result['content']:
                if isinstance(result['content'], dict):
                    content_text = result['content'].get('text', '') or result['content'].get('html', '')
                else:
                    content_text = str(result['content'])
                    
                if content_text:
                    st.download_button(
                        label="📥 원본 텍스트 파일 다운로드",
                        data=content_text,
                        file_name=f"extracted_text_{st.session_state.get('uploaded_file_name', 'unknown')}.txt",
                        mime="text/plain"
                    )

if __name__ == "__main__":
    main() 