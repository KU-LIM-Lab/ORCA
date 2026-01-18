import numpy as np
from typing import Dict, List, Optional, Any

import redis
from .settings import REDIS_CONFIG

from redisvl.schema import IndexSchema
from redisvl.index import SearchIndex
from redisvl.query import VectorQuery
from redisvl.utils.vectorize import OpenAITextVectorizer

import logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("faiss").setLevel(logging.WARNING)

from pathlib import Path
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parents[1] / ".env")


class VectorStore:
    def __init__(self, db_id:str, index_schema=None, redis_config: Dict[str, Any] = None):
        self.redis_config = redis_config or REDIS_CONFIG
        self.redis_client = redis.Redis(**self.redis_config)
        self.db = db_id

        self.schema = index_schema or IndexSchema.from_dict({
            "index": {
                "name": f"{self.db}_table_columns_index", # Remove
                "prefix": f"{self.db}:vector_doc",
                "key_separator": ":",
                "storage_type": "hash"
            },
            "fields": [
                {"name": "text", "type": "text"},
                {
                    "name": "embedding",
                    "type": "vector",
                    "attrs": {
                        "algorithm": "HNSW",
                        "dims": 1536,
                        "distance_metric": "COSINE",
                        "datatype": "float32"
                    }
                },
                {"name": "table", "type": "tag"}
            ]
        })

        self.index = SearchIndex(schema=self.schema, redis_client=self.redis_client)
        self.vectorizer = OpenAITextVectorizer(model="text-embedding-ada-002")

    def initialize(self):
        if not self.index.exists():
            self.index.create(overwrite=True)
            print("✅ Vector index created.")
        else:
            print("✅ Vector index already exists.")

    def upsert(self, table_name: str, doc_text: str, id=None) -> dict:
        embedding = self.vectorizer.embed(doc_text)
        embedding_bytes = np.array(embedding, dtype=np.float32).tobytes()

        return {
            "id": id or f"{self.db}:vector_doc:{table_name}",
            "text": doc_text,
            "embedding": embedding_bytes,
            "table": table_name
        }

    def load(self, docs: list[dict], id_field: str = "table") -> None:
        if docs:
            self.index.load(docs, id_field=id_field)
            print(f"📦 {len(docs)} vector documents loaded into Redis index.")

    def query(self, query_text: str, k: int = 10) -> list[dict]:
        embedding = self.vectorizer.embed(query_text)
        vector_query = VectorQuery(
            vector=embedding,
            vector_field_name="embedding",
            num_results=k,
            return_fields=["text", "table"]
        )
        return self.index.query(vector_query)