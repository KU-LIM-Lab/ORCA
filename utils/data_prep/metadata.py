import json
from typing import Dict, Optional
from prompts.data_prep_prompts import prompt, parser
from .related_tables import extract_schema
from utils.llm import call_llm
from utils.redis_client import redis_client
from utils.vectordb import VectorStore

# Generate table structure markdown (for LLM input)
def generate_table_markdown(sub_schema: Dict[str, dict]) -> str:
    lines = []
    for table, info in sub_schema.items():
        lines.append(f"# Table: {table}")
        for col, meta in info["columns"].items():
            parts = [f"{col} ({meta['type']})"]

            # Basic attributes
            if meta.get("pk"): parts.append("PK")
            if meta.get("fk"): parts.append(f"FK to {meta['fk']}")
            if meta.get("unique"): parts.append("UNIQUE")
            if meta.get("nullable") is False: parts.append("NOT NULL")
            if meta.get("default") is not None: parts.append(f"DEFAULT {meta['default']}")

            # Statistical information
            if "count" in meta: parts.append(f"count={meta['count']}")
            if "nulls" in meta: parts.append(f"nulls={meta['nulls']}")
            if "distinct" in meta: parts.append(f"distinct={meta['distinct']}")
            if "min" in meta and "max" in meta:
                parts.append(f"range={meta['min']}~{meta['max']}")
            if "avg" in meta: parts.append(f"avg={meta['avg']}")
            if "stddev" in meta: parts.append(f"stddev={meta['stddev']}")
            if "median(q2)" in meta: parts.append(f"median={meta['median(q2)']}")
            if "iqr_outlier_count" in meta:
                parts.append(f"outlier count based on IQR={meta['iqr_outlier_count']}")

            lines.append("- " + ", ".join(parts))

            # top_values, examples
            if "top_values" in meta and "values" in meta["top_values"]:
                top = ", ".join([f'{item["value"]}(count: {item["freq"]})' for item in meta["top_values"]["values"]])
                parts.append(f"top_values={top}")
            if "examples" in meta:
                examples = ", ".join(meta["examples"])
                parts.append(f"examples=[{examples}]")
            lines.append("- " + ", ".join(parts))

        for check in info.get("check_constraints", []):
            lines.append(f"- CHECK: {check}")
        for fk in info.get("foreign_keys", []):
            lines.append(f"- FOREIGN KEY: {fk[0]} → {fk[1]}.{fk[2]}")
        lines.append("")
    return "\n".join(lines)


def generate_metadata(table_name: str, schema: dict) -> dict:
    schema_md = generate_table_markdown({table_name: schema})
    result = call_llm(prompt=prompt, parser=parser, variables={"schema": schema_md})
    metadata = result.model_dump()
    metadata["schema"] = schema
    return metadata


def update_metadata(db_id, schema: Optional[dict] = None, change: bool = False) -> bool:
    vector_store = VectorStore(db_id)
    vector_store.initialize()

    if schema is None:
        schema = extract_schema(db_id)
    
    if not change:
        print("No metadata update is needed.")
        return False

    docs_to_load = []

    for table_name, table_schema in schema.items():
        stored = redis_client.get(f"{db_id}:metadata:{table_name}")
        stored_metadata = json.loads(stored) if stored else {}

        if table_schema != stored_metadata.get("schema"):
            new_metadata = generate_metadata(table_name, table_schema)
            redis_client.set(f"{db_id}:metadata:{table_name}", json.dumps(new_metadata, indent=2, default=str))
            redis_client.sadd(f"{db_id}:metadata:table_names", table_name)
            print(f"✅ Metadata updated for table: {table_name}")

            doc_text = f"{new_metadata.get('description', '')}\n\nColumns:\n"
            doc_text += "\n".join([f"- `{col}`: {desc}" for col, desc in new_metadata["columns"].items()])
            
            doc = vector_store.upsert(table_name, doc_text)
            docs_to_load.append(doc)
        else:
            print(f"🔁 No metadata change for table: {table_name}")

    if docs_to_load:
        vector_store.load(docs_to_load)  # Actually store in Redis vector index
        return True
    
    return False


# updated_docs = []
# async def process_table_metadata(table_name: str, table_schema: dict) -> Optional[dict]:
#     new_metadata, stored = await asyncio.gather(
#         generate_metadata(table_name, table_schema),
#         cache.get_async(f"metadata:{table_name}")
#     )
#     stored_metadata = json.loads(stored) if stored else {}

#     if new_metadata["columns"] != stored_metadata.get("columns"):
#         await cache.set_async(
#             f"metadata:{table_name}",
#             json.dumps(new_metadata, indent=2, default=str)
#         )
#         await cache.sadd_asyc("metadata:table_names", table_name)
#         print(f"Metadata updated for table: {table_name}")

#         # Generate vector document
#         doc_text = f"{new_metadata.get('description', '')}\n\nColumns:\n"
#         doc_text += "\n".join([f"- `{col}`: {desc}" for col, desc in new_metadata["columns"].items()])
#         return await vector_store.upsert(table_name, doc_text)
#     else:
#         print(f"No metadata change for table: {table_name}")
#         return None

# async def update_metadata(schema: Optional[dict] = None, change: bool = False) -> bool:
#     vector_store = VectorStore()
#     await vector_store.initialize()

#     if schema is None:
#         schema = await extract_schema()
        
#     if not change:
#         print("No metadata update is needed.")
#         return False

#     tasks = [
#         process_table_metadata(table_name, table_schema)
#         for table_name, table_schema in schema.items()
#     ]
#     results = await asyncio.gather(*tasks)

#     # Store excluding None
#     docs_to_store = [doc for doc in results if doc is not None]
#     if docs_to_store:
#         await vector_store.load(docs_to_store, id_field="table")
#         print(f"vector documents stored in Redis!")
#         return True
#     return False