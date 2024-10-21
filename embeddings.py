from transformers import BertTokenizer, BertModel
import torch
from pinecone_text.sparse import BM25Encoder
from openai import OpenAI
from os import getenv
import numpy as np


class EmbeddingGenerator:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.model = BertModel.from_pretrained("bert-base-uncased")

    def generate_dense_vector(self, article: str) -> list[float]:
        encoding = self.tokenizer.batch_encode_plus(
            [article],
            padding=True,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=True,
        )
        input_ids = encoding["input_ids"]

        attention_mask = encoding["attention_mask"]

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            word_embeddings = outputs.last_hidden_state

            sentence_embedding = word_embeddings.mean(dim=1).flatten().tolist()

        return sentence_embedding

    def generate_sparse_vectors(self, encoder: BM25Encoder, article: str):
        return encoder.encode_documents(article)

    def generate_sparse_query(self, encoder: BM25Encoder, query: str):
        return encoder.encode_queries(query)
