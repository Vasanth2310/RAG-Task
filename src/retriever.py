from typing import List, Dict, Any

class RAGRetriever:
    def __init__(self, vector_store, embedding_manager):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(self, query: str, top_k: int = 5, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        query_embedding = self.embedding_manager.generate_embeddings([query])[0]
        results = self.vector_store.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )
        retrieved_docs = []
        if results.get('documents') and results['documents'][0]:
            docs = results['documents'][0]
            metadatas = results['metadatas'][0]
            distances = results['distances'][0]
            ids = results['ids'][0]
            for i, (doc_id, document, metadata, distance) in enumerate(zip(ids, docs, metadatas, distances)):
                similarity_score = 1 - distance
                if similarity_score >= score_threshold:
                    retrieved_docs.append({
                        'id': doc_id,
                        'content': document,
                        'metadata': metadata,
                        'similarity_score': similarity_score,
                        'distance': distance,
                        'rank': i + 1
                    })
        return retrieved_docs

def reflect_answer(query: str, answer: str, retrieved_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
    max_score = max([d['similarity_score'] for d in retrieved_docs], default=0.0)
    contains_query_tokens = any(tok.lower() in answer.lower() for tok in query.split()[:6])
    relevance = "HIGH" if (max_score >= 0.3 and contains_query_tokens) else ("MEDIUM" if max_score >= 0.15 else "LOW")
    return {"max_similarity": float(max_score), "contains_query_tokens_in_answer": contains_query_tokens, "relevance": relevance}
