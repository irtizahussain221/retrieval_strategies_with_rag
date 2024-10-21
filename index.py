from dotenv import load_dotenv
from articles import get_articles
from pinecone_text.sparse import BM25Encoder
from embeddings import EmbeddingGenerator
from article_embedder import (
    embed_articles,
    INDEX_NAME,
    DENSE_VECTOR_NAMESPACE,
    HYBRID_VECTOR_NAMESPACE,
)
from queries import SAMPLE_QUERIES
# import pickle
from vector_store import PineconeStore

load_dotenv()

# articles = get_articles()
# bm25 = BM25Encoder()
# bm25_encoder = bm25.fit(articles)
# with open("bm25_model.pkl", "wb") as f:
#     pickle.dump(bm25_encoder, f)

# with open("bm25_model.pkl", "rb") as f:
#     bm25_encoder: BM25Encoder = pickle.load(f)

# embed_articles(articles, 
            #    bm25_encoder
            #    )


# Embed the query
embedder = EmbeddingGenerator()
# (simple_query, query_with_hybrid) = SAMPLE_QUERIES[3]
# query_dense_vector = embedder.generate_dense_vector(simple_query)
# query_sparse_vector = embedder.generate_sparse_query(bm25_encoder, simple_query)
# query_dense_vector_with_keywords = embedder.generate_dense_vector(query_with_hybrid)
# query_sparse_vector_with_keywords = embedder.generate_sparse_query(
#     bm25_encoder, query_with_hybrid
# )
query = f"""Who was selected as chairman after Rotterdam port cargo dispute?
"""
query_dense_vector = embedder.generate_dense_vector(query)
# query_sparse_vector = embedder.generate_sparse_query(bm25_encoder, query)

dense_vector_store = PineconeStore(
    index_name=INDEX_NAME, namespace=DENSE_VECTOR_NAMESPACE
)
# hybrid_vector_store = PineconeStore(
#     index_name=INDEX_NAME, namespace=HYBRID_VECTOR_NAMESPACE
# )

# response_hybrid = hybrid_vector_store.hybrid_vector_query(
#     query_dense_vector, 
#     query_sparse_vector
# )
# print("\n\n\n\n\n", response_hybrid)

response_dense = dense_vector_store.dense_vector_query(query_dense_vector)
# response_hybrid = hybrid_vector_store.hybrid_vector_query(
#     query_dense_vector, query_sparse_vector
# )
print("\n\n\n\n\n simple", response_dense)

# response_dense = dense_vector_store.dense_vector_query(query_dense_vector_with_keywords)
# response_hybrid = hybrid_vector_store.hybrid_vector_query(
#     query_dense_vector_with_keywords, query_sparse_vector_with_keywords
# )
# print("\n\n\n\n\n hybrid with keywords", response_hybrid)
