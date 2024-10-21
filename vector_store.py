from pinecone import Pinecone, Vector
from dotenv import load_dotenv
from os import getenv
from dataclasses import dataclass


@dataclass
class Document:
    id: str
    text: str
    dense_vector: list[float]
    sparse_values: dict


load_dotenv()

VECTOR_STORE_DIMENSIONS = 768


class PineconeStore:
    def __init__(self, index_name: str, namespace: str):

        self.client = Pinecone(api_key=getenv("VECTOR_STORE_API_KEY"))
        self.index = self.client.Index(index_name)
        self.namespace = namespace

    def upsert_dense_vectors(self, documents: list[Document]):
        vectors = [
            Vector(
                id=document.id,
                values=document.dense_vector,
                metadata={"text": document.text},
            )
            for document in documents
        ]
        self.index.upsert(vectors=vectors, namespace=self.namespace)

    def upsert_hybrid_vectors(self, documents: list[Document]):
        vectors = [
            {
                "id": document.id,
                "values": document.dense_vector,
                "sparse_values": document.sparse_values,
                "metadata": {"text": document.text},
            }
            for document in documents
        ]
        self.index.upsert(vectors=vectors, namespace=self.namespace)

    def dense_vector_query(self, query_vector: list[float], top_k: int = 10):
        return self.index.query(
            vector=query_vector,
            top_k=20,
            namespace=self.namespace,
            include_metadata=False,
        )

    def hybrid_vector_query(
        self, query_vector: list[float], 
        sparse_vector: dict,
        top_k: int = 10
    ):
        # hdense, hsparse = hybrid_score_norm(query_vector, sparse_vector, alpha=0.30)
        return self.index.query(
            vector=query_vector,
            top_k=20,
            namespace=self.namespace,
            include_metadata=False,
            sparse_vector=sparse_vector,
        )

def hybrid_score_norm(dense, sparse, alpha: float):
    """Hybrid score using a convex combination

    alpha * dense + (1 - alpha) * sparse

    Args:
        dense: Array of floats representing
        sparse: a dict of `indices` and `values`
        alpha: scale between 0 and 1
    """
    if alpha < 0 or alpha > 1:
        raise ValueError("Alpha must be between 0 and 1")
    hs = {
        'indices': sparse['indices'],
        'values':  [v * (1 - alpha) for v in sparse['values']]
    }
    return [v * alpha for v in dense], hs