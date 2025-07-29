from opensearchpy import OpenSearch
from typing import List, Dict, Any, Optional
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from file.search import create_opensearch_client, search_chunks, vector_search_chunks, hybrid_search_chunks
from file.upstage import create_embeddings_client
from langchain_openai import AzureChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()


def rag_search(
    query: str,
    client: Optional[OpenSearch] = None,
    search_type: str = "hybrid",
    size: int = 5
) -> Dict[str, Any]:
    """
    OpenSearch를 사용한 RAG 기반 검색을 수행합니다.
    
    Args:
        query: 검색 쿼리
        client: OpenSearch 클라이언트 (None이면 새로 생성)
        search_type: 검색 타입 ("text", "vector", "hybrid")
        size: 반환할 결과 수
        
    Returns:
        검색 결과와 관련 메타데이터
    """
    if client is None:
        client = create_opensearch_client()
    
    # 검색 수행
    search_results = []
    
    if search_type == "text":
        search_results = search_chunks(client, query, size=size)
    elif search_type == "vector":
        # 쿼리 임베딩 생성
        embeddings_client = create_embeddings_client()
        if embeddings_client:
            try:
                query_embedding = embeddings_client.embed_query(query)
                search_results = vector_search_chunks(client, query_embedding, size=size)
            except Exception as e:
                print(f"벡터 검색 실패, 텍스트 검색으로 대체: {e}")
                search_results = search_chunks(client, query, size=size)
        else:
            search_results = search_chunks(client, query, size=size)
    elif search_type == "hybrid":
        # 하이브리드 검색을 위한 쿼리 임베딩 생성
        embeddings_client = create_embeddings_client()
        query_vector = None
        if embeddings_client:
            try:
                query_vector = embeddings_client.embed_query(query)
            except Exception as e:
                print(f"임베딩 생성 실패, 텍스트 검색만 사용: {e}")
        
        search_results = hybrid_search_chunks(client, query, query_vector, size=size)
    
    return {
        "query": query,
        "search_results": search_results,
        "total_results": len(search_results)
    }

def get_context_from_results(results: List[Dict[str, Any]], max_context_length: int = 50000) -> str:
    """
    검색 결과를 컨텍스트 문자열로 변환합니다.
    
    Args:
        results: 검색 결과 리스트
        max_context_length: 최대 컨텍스트 길이
        
    Returns:
        컨텍스트 문자열
    """
    context_parts = []
    current_length = 0
    
    for result in results:
        content = result.get("content", "")
        source = f"[출처: {result.get('document_name', 'unknown')}]"
        
        # 단일 청크가 max_context_length를 초과하는 경우 잘라서 포함
        if len(content) > max_context_length:
            content = content[:max_context_length-len(source)-10] + "..."
        
        formatted_content = f"{source}\n{content}"
        
        if current_length + len(formatted_content) > max_context_length:
            # 남은 공간이 있으면 부분적으로라도 포함
            remaining_space = max_context_length - current_length
            if remaining_space > len(source) + 50:  # 최소한의 내용이 들어갈 공간이 있으면
                truncated_content = content[:remaining_space-len(source)-10] + "..."
                context_parts.append(f"{source}\n{truncated_content}")
            break
        
        context_parts.append(formatted_content)
        current_length += len(formatted_content)
    
    return "\n\n".join(context_parts)

def rag_query(
    question: str,
    client: Optional[OpenSearch] = None,
    search_type: str = "hybrid",
    context_size: int = 5,
    max_context_length: int = 50000
) -> Dict[str, Any]:
    """
    RAG를 사용한 질의응답을 수행합니다.
    
    Args:
        question: 질문
        client: OpenSearch 클라이언트
        search_type: 검색 타입
        context_size: 컨텍스트로 사용할 검색 결과 수
        max_context_length: 최대 컨텍스트 길이
        
    Returns:
        질문, 컨텍스트, 검색 메타데이터를 포함한 딕셔너리
    """
    # RAG 검색 수행
    rag_result = rag_search(
        query=question,
        client=client,
        search_type=search_type,
        size=context_size
    )
    
    # 컨텍스트 생성
    context = get_context_from_results(
        rag_result["search_results"],
        max_context_length=max_context_length
    )
    
    return {
        "question": question,
        "context": context,
        "search_metadata": {
            "total_results": rag_result["total_results"]
        },
        "search_results": rag_result["search_results"]
    }

def create_llm_client():
    """Azure OpenAI LLM 클라이언트를 생성합니다."""
    try:
        llm = AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),  # AZURE_OPENAI_DEPLOYMENT 사용
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
            temperature=0.1
        )
        return llm
    except Exception as e:
        print(f"⚠️ LLM 클라이언트 생성 실패: {e}")
        return None

def generate_answer_with_llm(question: str, context: str, llm_client=None) -> str:
    """
    컨텍스트를 바탕으로 LLM을 사용해 질문에 대한 답변을 생성합니다.
    
    Args:
        question: 질문
        context: 검색된 컨텍스트
        llm_client: LLM 클라이언트 (None이면 새로 생성)
        
    Returns:
        생성된 답변
    """
    if not context:
        return "관련된 정보를 찾을 수 없어 답변을 생성할 수 없습니다."
    
    if llm_client is None:
        llm_client = create_llm_client()
        if llm_client is None:
            return "LLM 서비스에 연결할 수 없어 답변을 생성할 수 없습니다."
    
    prompt = f"""Please provide an answer to the question based on the following context.

Context:
{context}

Question: {question}

Please follow this format when answering:
1. First, quote the exact relevant text from the context in its original language (English)
2. Then provide a Korean translation/explanation of that content
3. Structure your answer like this:
   - Start with the exact quote from the document
   - Then add: "주어진 문서에서 [topic]에 대한 정보는 위와 같습니다."
   - Follow with a Korean explanation/translation
4. write page number in the context
Guidelines:
- Use only the information contained in the context
- Quote the exact text that answers the question
- Provide accurate Korean translation
- If no relevant information is found, respond with "제공된 문서에서 해당 정보를 찾을 수 없습니다"

Answer:"""

    try:
        response = llm_client.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        print(f"⚠️ 답변 생성 중 오류: {e}")
        return f"답변 생성 중 오류가 발생했습니다: {str(e)}"

# 테스트 함수
def main():
    """RAG 시스템 테스트 함수"""
    client = create_opensearch_client()
    llm_client = create_llm_client()
    
    # 샘플 질문들
    sample_questions = [
        # "What type of gate valve is required for ASME class 1500 and higher according to the specification?",
        # "Are plastic valve handwheels acceptable under this specification?",
        # "According to the 'General' section, what is the primary responsibility of the Supplier regarding their equipment and accessories, and what specific aspects should their equipment incorporate?",
        # "What is the minimum operational history in an industrial plant required for the equipment offered by the Supplier?",
        # "Before initiating fabrication work, what crucial approval must the Supplier obtain, and what must be completed at the factory prior to shipment?",
        "FORGING",
        "about CASTING",
        "Proposal Requirements"
    ]
    
    for question in sample_questions:
        print(f"\n질문: {question}")
        print("-" * 80)
        
        result = rag_query(question, client)
        
        if result['context']:
            # LLM을 사용해 답변 생성
            answer = generate_answer_with_llm(question, result['context'], llm_client)
            print(f"\n💬 AI 답변:\n{answer}")
            
            # 참고한 청크 정보 표시 (컨텍스트에 실제 포함된 청크들)
            chunk_ids = [str(res.get('chunk_id', 'unknown')) for res in result['search_results'] if res.get('content')]
            print(f"\n📄 참고한 청크: {', '.join(chunk_ids[:5])}")  # 상위 5개 표시
            
            # 디버깅 정보 (선택적)
            print(f"\n📊 검색 메타데이터:")
            print(f"  - 총 검색 결과 수: {result['search_metadata']['total_results']}")
            
        else:
            print("\n❌ 관련된 컨텍스트를 찾을 수 없습니다.")

if __name__ == "__main__":
    main()