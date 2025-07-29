from opensearchpy import OpenSearch
from typing import List, Dict, Any, Optional
import hashlib
import json
from datetime import datetime
import urllib3
import os  # 이 줄 추가
from dotenv import load_dotenv

load_dotenv()

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# local
# def create_opensearch_client() -> OpenSearch:
#     """OpenSearch 클라이언트를 생성합니다."""
#     return OpenSearch(
#         hosts=[{
#             'host': os.getenv('OPENSEARCH_HOST', 'localhost'), 
#             'port': int(os.getenv('OPENSEARCH_PORT', '9200'))
#         }],
#         http_auth=(
#             os.getenv('OPENSEARCH_USERNAME', 'admin'),
#             os.getenv('OPENSEARCH_PASSWORD')  # 기본값 제거
#         ),
#         use_ssl=os.getenv('OPENSEARCH_USE_SSL', 'true').lower() == 'true',
#         verify_certs=False
#     )

# aws
def create_opensearch_client() -> OpenSearch:
    """OpenSearch 클라이언트를 생성합니다."""
    username = os.getenv('OPENSEARCH_USERNAME')
    password = os.getenv('OPENSEARCH_PASSWORD')
    http_auth = (username, password) if username and password else None

    return OpenSearch(
        hosts=[{
            'host': os.getenv('OPENSEARCH_HOST', 'localhost'),
            'port': int(os.getenv('OPENSEARCH_PORT', '9200'))
        }],
        http_auth=http_auth,
        use_ssl=os.getenv('OPENSEARCH_USE_SSL', 'true').lower() == 'true',
        verify_certs=False  # 테스트 시 False, 운영에서는 True
    )



def reset_index_with_embeddings(client: OpenSearch, index_name: str = "document-chunks"):
    """
    기존 인덱스를 삭제하고 임베딩 지원이 포함된 새 인덱스를 생성합니다.
    
    Args:
        client: OpenSearch 클라이언트
        index_name: 인덱스 이름
    """
    try:
        # 기존 인덱스가 있으면 삭제
        if client.indices.exists(index=index_name):
            print(f"🗑️ 기존 인덱스 {index_name} 삭제 중...")
            client.indices.delete(index=index_name)
            print(f"✅ 기존 인덱스 {index_name} 삭제 완료")
        
        # 새로운 인덱스 설정 (knn_vector 타입 사용)
        index_settings = {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 1,
                "index.knn": True  # KNN 플러그인 활성화
            },
            "mappings": {
                "properties": {
                    "chunk_id": {"type": "integer"},
                    "content": {"type": "text"},
                    "document_name": {"type": "keyword"},
                    "timestamp": {"type": "date"},
                    "metadata": {"type": "object"},
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": 1536,
                        "method": {
                            "name": "hnsw",
                            "space_type": "cosinesimil",
                            "engine": "lucene"
                        }
                    }
                }
            }
        }
        
        # 새 인덱스 생성
        print(f"🔄 새 인덱스 {index_name} 생성 중...")
        client.indices.create(index=index_name, body=index_settings)
        print(f"✅ 새 인덱스 {index_name} 생성 완료!")
        
        return True
        
    except Exception as e:
        print(f"❌ 인덱스 재설정 실패: {e}")
        return False

def save_chunks_to_opensearch(
    chunks: List[str], 
    client: OpenSearch,
    document_name: str,
    metadata: Optional[Dict[str, Any]] = None,
    embeddings: Optional[List[List[float]]] = None,
    index_name: str = "document-chunks"
) -> List[str]:
    """
    Upstage에서 생성된 청크들을 OpenSearch에 저장합니다.
    
    Args:
        chunks: 저장할 청크 리스트
        client: OpenSearch 클라이언트
        document_name: 문서 이름
        metadata: 추가 메타데이터
        embeddings: 청크에 대응하는 임베딩 벡터 리스트
        index_name: 인덱스 이름
        
    Returns:
        저장된 문서 ID 목록
    """
    # 인덱스가 없다면 생성
    if not client.indices.exists(index=index_name):
        index_settings = {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 1,
                "index.knn": True  # KNN 플러그인 활성화
            },
            "mappings": {
                "properties": {
                    "chunk_id": {"type": "integer"},
                    "content": {"type": "text"},
                    "document_name": {"type": "keyword"},
                    "timestamp": {"type": "date"},
                    "metadata": {"type": "object"},
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": 1536,
                        "method": {
                            "name": "hnsw",
                            "space_type": "cosinesimil",
                            "engine": "lucene"
                        }
                    }
                }
            }
        }
        client.indices.create(index=index_name, body=index_settings)
        print(f"✅ 인덱스 생성됨: {index_name}")
    
    saved_ids = []
    timestamp = datetime.now().isoformat()
    
    # 각 청크를 저장
    for i, chunk in enumerate(chunks):
        # 문서 ID 생성 (문서명 + 청크 인덱스의 해시)
        doc_id = hashlib.md5(f"{document_name}_chunk_{i}".encode()).hexdigest()
        
        doc = {
            "chunk_id": i,
            "content": chunk.strip(),
            "document_name": document_name,
            "timestamp": timestamp,
            "metadata": metadata or {}
        }
        
        # 임베딩이 있으면 추가
        if embeddings and i < len(embeddings):
            doc["embedding"] = embeddings[i]
        
        try:
            response = client.index(
                index=index_name, 
                id=doc_id,
                body=doc, 
                refresh=True
            )
            saved_ids.append(doc_id)
            print(f"📦 저장됨: chunk {i} -> {doc_id}")
        except Exception as e:
            print(f"❌ 청크 {i} 저장 실패: {str(e)}")
    
    print(f"✅ 총 {len(saved_ids)}개의 청크가 저장되었습니다.")
    return saved_ids

def search_chunks(
    client: OpenSearch,
    query: str,
    index_name: str = "document-chunks",
    size: int = 10
) -> List[Dict[str, Any]]:
    """
    OpenSearch에서 청크를 검색합니다.
    
    Args:
        client: OpenSearch 클라이언트
        query: 검색 쿼리
        index_name: 인덱스 이름
        size: 반환할 결과 수
        
    Returns:
        검색 결과 목록
    """
    search_body = {
        "query": {
            "match": {
                "content": query
            }
        },
        "size": size,
        "_source": ["chunk_id", "content", "document_name", "timestamp", "metadata"]
    }
    
    try:
        response = client.search(index=index_name, body=search_body)
        hits = response['hits']['hits']
        
        results = []
        for hit in hits:
            result = {
                "id": hit["_id"],
                "score": hit["_score"],
                **hit["_source"]
            }
            results.append(result)
        
        return results
    except Exception as e:
        print(f"❌ 검색 실패: {str(e)}")
        return []

def vector_search_chunks(
    client: OpenSearch,
    query_vector: List[float],
    index_name: str = "document-chunks",
    size: int = 10,
    min_score: float = 0.7
) -> List[Dict[str, Any]]:
    """
    벡터 유사도를 이용해 OpenSearch에서 청크를 검색합니다.
    
    Args:
        client: OpenSearch 클라이언트
        query_vector: 검색할 벡터
        index_name: 인덱스 이름
        size: 반환할 결과 수
        min_score: 최소 유사도 점수
        
    Returns:
        검색 결과 목록
    """
    search_body = {
        "query": {
            "knn": {
                "embedding": {
                    "vector": query_vector,
                    "k": size
                }
            }
        },
        "min_score": min_score,
        "size": size,
        "_source": ["chunk_id", "content", "document_name", "timestamp", "metadata"]
    }
    
    try:
        response = client.search(index=index_name, body=search_body)
        hits = response['hits']['hits']
        
        results = []
        for hit in hits:
            result = {
                "id": hit["_id"],
                "score": hit["_score"],
                **hit["_source"]
            }
            results.append(result)
        
        return results
    except Exception as e:
        print(f"❌ 벡터 검색 실패: {str(e)}")
        return []

def hybrid_search_chunks(
    client: OpenSearch,
    query_text: str,
    query_vector: Optional[List[float]] = None,
    index_name: str = "document-chunks",
    size: int = 10,
    text_weight: float = 0.5,
    vector_weight: float = 0.5
) -> List[Dict[str, Any]]:
    """
    텍스트 검색과 벡터 검색을 결합한 하이브리드 검색을 수행합니다.
    
    Args:
        client: OpenSearch 클라이언트
        query_text: 검색할 텍스트
        query_vector: 검색할 벡터 (선택적)
        index_name: 인덱스 이름
        size: 반환할 결과 수
        text_weight: 텍스트 검색 가중치
        vector_weight: 벡터 검색 가중치
        
    Returns:
        검색 결과 목록
    """
    if query_vector is None:
        # 벡터가 없으면 텍스트 검색만 수행
        return search_chunks(client, query_text, index_name, size)
    
    search_body = {
        "query": {
            "bool": {
                "should": [
                    {
                        "match": {
                            "content": {
                                "query": query_text,
                                "boost": text_weight
                            }
                        }
                    },
                    {
                        "knn": {
                            "embedding": {
                                "vector": query_vector,
                                "k": size,
                                "boost": vector_weight
                            }
                        }
                    }
                ]
            }
        },
        "size": size,
        "_source": ["chunk_id", "content", "document_name", "timestamp", "metadata"]
    }
    
    try:
        response = client.search(index=index_name, body=search_body)
        hits = response['hits']['hits']
        
        results = []
        for hit in hits:
            result = {
                "id": hit["_id"],
                "score": hit["_score"],
                **hit["_source"]
            }
            results.append(result)
        
        return results
    except Exception as e:
        print(f"❌ 하이브리드 검색 실패: {str(e)}")
        return []