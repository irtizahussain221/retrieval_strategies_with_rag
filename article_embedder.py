from embeddings import EmbeddingGenerator
from vector_store import PineconeStore, Document
from pinecone_text.sparse import BM25Encoder


INDEX_NAME = "articles"
DENSE_VECTOR_NAMESPACE = "dense_vector_articles"
HYBRID_VECTOR_NAMESPACE = "hybrid_vector_articles"


def embed_articles(articles: list[str], 
                #    bm25_encoder: BM25Encoder
                   ):
    embedder = EmbeddingGenerator()
    dense_vector_documents = []
    hybrid_vector_documents = []

    for index, article in enumerate(articles):
        dense_vector_documents.append(
            Document(
                id=f"article-{index}",
                text=article,
                dense_vector=embedder.generate_dense_vector(article),
                sparse_values=[],
            )
        )

        # hybrid_vector_documents.append(
        #     Document(
        #         id=f"article-{index}",
        #         text=article,
        #         dense_vector=embedder.generate_dense_vector(article),
        #         sparse_values=embedder.generate_sparse_vectors(bm25_encoder, article),
        #     )
        # )

    dense_vector_store = PineconeStore(
        index_name=INDEX_NAME, namespace=DENSE_VECTOR_NAMESPACE
    )
    # hybrid_vector_store = PineconeStore(
    #     index_name=INDEX_NAME, namespace=HYBRID_VECTOR_NAMESPACE
    # )

    dense_vector_store.upsert_dense_vectors(dense_vector_documents)
    # hybrid_vector_store.upsert_hybrid_vectors(hybrid_vector_documents)
