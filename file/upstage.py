# pip install requests beautifulsoup4 langchain-openai

import requests
import os
import re
from typing import Optional, Dict, Any, List, ClassVar
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from langchain_openai import AzureOpenAIEmbeddings
load_dotenv()


def extract_chunks_from_html(html_content: str, use_dynamic_headers: bool = True) -> List[str]:
    """
    HTML 컨텐츠를 청크 헤더 기준으로 분할합니다.
    
    Args:
        html_content: HTML 문자열
        use_dynamic_headers: True면 자동으로 "숫자.숫자" 패턴 감지, False면 기본 헤더 사용
        
    Returns:
        분할된 청크 목록
    """
    # BeautifulSoup으로 파싱
    soup = BeautifulSoup(html_content, "html.parser")
    
    chunks = []
    current_chunk = ""
    collecting = False
    current_page = None
    
    # 정규표현식 패턴: "숫자.숫자" 형태로 시작하는 모든 섹션 (서브섹션 포함)
    # 예: "1.1", "1.2", "1.3", "2.1", "2.2" 등 모두 매치
    section_pattern = re.compile(r'^\d+\.\d+(\s|$)')
    
    # 모든 요소를 순서대로 처리
    all_elements = soup.find_all(['h1', 'h2', 'h3', 'p', 'footer'])
    
    for tag in all_elements:
        text = tag.get_text(strip=True)
        
        # 페이지 번호 감지
        is_footer = tag.name == 'footer'
        is_page_number = False
        
        # footer 태그이거나 단순 숫자만 있는 경우 페이지 번호로 판단
        if is_footer and text and text.isdigit():
            is_page_number = True
            current_page = text
            continue  # 페이지 번호는 별도 처리하고 계속
        elif text and text.isdigit() and len(text) <= 3 and tag.name == 'p':
            # p 태그에서 단순 숫자인 경우도 페이지 번호로 간주
            is_page_number = True
            current_page = text
            continue
        
        # 빈 텍스트 건너뛰기
        if not text:
            continue
        
        # 동적 헤더 감지: "숫자.숫자" 패턴으로 시작하는 텍스트
        is_section_header = False
        if use_dynamic_headers:
            is_section_header = section_pattern.match(text)
        else:
            # 기본 헤더 목록 사용 (fallback)
            default_headers = ["1.0 GENERAL", "2.0 SCOPE OF SUPPLY", "3.0", "4.0", "5.0"]
            is_section_header = any(text.startswith(header) for header in default_headers)
        
        # 새로운 청크 시작 조건
        if is_section_header:
            # 이전 청크에 페이지 정보 추가 후 저장
            if current_chunk:
                if current_page:
                    current_chunk += f"\n[페이지 {current_page}]"
                chunks.append(current_chunk.strip())
            # 새 청크 시작
            current_chunk = text
            collecting = True
        elif collecting:
            # 일반 텍스트 추가
            current_chunk += "\n" + text
    
    # 마지막 청크에 페이지 정보 추가 후 저장
    if current_chunk:
        if current_page:
            current_chunk += f"\n[페이지 {current_page}]"
        chunks.append(current_chunk.strip())
    
    return chunks

def create_embeddings_client():
    """Azure OpenAI 임베딩 클라이언트를 생성합니다."""
    try:
        embeddings = AzureOpenAIEmbeddings(
            azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"), 
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"), # Azure OpenAI 서비스의 실제 엔드포인트 URL
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
        )
        return embeddings
    except Exception as e:
        print(f"⚠️ 임베딩 클라이언트 생성 실패: {e}")
        return None

def generate_embeddings_for_chunks(chunks: List[str]) -> List[List[float]]:
    """
    청크 리스트를 임베딩 벡터로 변환합니다.
    
    Args:
        chunks: 임베딩할 텍스트 청크 리스트
        
    Returns:
        임베딩 벡터 리스트 (각 청크당 하나의 벡터)
    """
    if not chunks:
        return []
    
    try:
        embeddings_client = create_embeddings_client()
        if embeddings_client is None:
            print("❌ 임베딩 클라이언트를 생성할 수 없습니다.")
            return []
        
        print(f"🔄 {len(chunks)}개 청크에 대한 임베딩 생성 중...")
        
        # 청크를 배치로 처리하여 임베딩 생성
        embeddings = embeddings_client.embed_documents(chunks)
        
        print(f"✅ {len(embeddings)}개 임베딩 벡터 생성 완료")
        return embeddings
        
    except Exception as e:
        print(f"❌ 임베딩 생성 중 오류: {e}")
        return []

def process_document_with_upstage(
    file_path: str, 
    api_key: Optional[str] = None,
    ocr: str = "force",
    base64_encoding: str = "['table']",
    model: str = "document-parse"
) -> Dict[Any, Any]:
    """
    Upstage API를 사용하여 문서를 처리합니다.
    
    Args:
        file_path: 처리할 파일 경로
        api_key: Upstage API 키 (없으면 환경변수에서 가져옴)
        ocr: OCR 설정
        base64_encoding: Base64 인코딩 설정  
        model: 사용할 모델
        
    Returns:
        API 응답 결과
    """
    if api_key is None:
        api_key = os.getenv("UPSTAGE_API_KEY", "UPSTAGE_API_KEY")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")
    
    url = "https://api.upstage.ai/v1/document-digitization"
    headers = {"Authorization": f"Bearer {api_key}"}
    
    try:
        with open(file_path, "rb") as file:
            files = {"document": file}
            data = {
                "ocr": ocr, 
                "base64_encoding": base64_encoding, 
                "model": model
            }
            response = requests.post(url, headers=headers, files=files, data=data)
            response.raise_for_status()
            
            # API 응답 JSON 파싱
            result = response.json()
            
            # HTML 컨텐츠가 있으면 청크로 분할
            if 'content' in result and 'html' in result['content']:
                html_content = result['content']['html']
                try:
                    chunks = extract_chunks_from_html(html_content)
                    result['chunks'] = chunks
                    result['chunks_count'] = len(chunks)
                    
                    # 디버깅: 감지된 헤더들 추가
                    soup = BeautifulSoup(html_content, "html.parser")
                    section_pattern = re.compile(r'^\d+\.\d+(\s|$)')
                    detected_headers = []
                    
                    for tag in soup.find_all(['h1', 'h2', 'h3', 'p', 'footer']):
                        text = tag.get_text(strip=True)
                        if section_pattern.match(text):
                            detected_headers.append(text)
                    
                    result['detected_headers'] = detected_headers
                    
                    # 청크에 대한 임베딩 생성
                    if chunks:
                        try:
                            print("🔄 청크 임베딩 생성 시작...")
                            embeddings = generate_embeddings_for_chunks(chunks)
                            if embeddings:
                                result['embeddings'] = embeddings
                                result['embeddings_count'] = len(embeddings)
                                print(f"✅ {len(embeddings)}개 임베딩 벡터가 생성되었습니다.")
                            else:
                                result['embeddings'] = []
                                result['embeddings_count'] = 0
                                result['embeddings_error'] = "임베딩 생성 실패"
                        except Exception as embed_e:
                            print(f"⚠️ 임베딩 생성 중 오류: {embed_e}")
                            result['embeddings'] = []
                            result['embeddings_count'] = 0
                            result['embeddings_error'] = f"임베딩 생성 중 오류: {str(embed_e)}"
                    else:
                        result['embeddings'] = []
                        result['embeddings_count'] = 0
                    
                except Exception as e:
                    # 청크 분할 실패해도 원본 결과는 반환
                    result['chunks_error'] = f"청크 분할 중 오류: {str(e)}"
                    result['chunks'] = []
                    result['embeddings'] = []
                    result['embeddings_count'] = 0
            else:
                result['chunks'] = []
                result['chunks_count'] = 0
                result['embeddings'] = []
                result['embeddings_count'] = 0
            
            return result
            
    except requests.exceptions.RequestException as e:
        raise Exception(f"API 요청 중 오류가 발생했습니다: {str(e)}")
