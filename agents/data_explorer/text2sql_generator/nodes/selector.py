import json, re
from typing import Dict, Any, List
from ORCA.prompts.text2sql_generator_prompts import selector_template
from utils.redis_client import redis_client
from utils.llm import call_llm
from langchain_core.language_models.chat_models import BaseChatModel

from utils.vectordb import VectorStore

def extract_metadata(table_name: str, meta: Dict[str, Any]) -> Dict[str, Any]:
    schema = meta.get("schema", {})
    columns_schema = schema.get("columns", {})
    columns_desc = meta.get("columns", {})

    columns = [{
        "name": col,
        "type": col_schema.get("type"),
        "desc": columns_desc.get(col, ""),
        "nullable": col_schema.get("nullable", False),
        "fk": col_schema.get("fk"),
        "examples": col_schema.get("examples", []),
        "min": col_schema.get("min"),
        "max": col_schema.get("max")
    } for col, col_schema in columns_schema.items()]

    return {
        "table_name": table_name,
        "description": meta.get("description", ""),
        "columns": columns,
        "foreign_keys": schema.get("foreign_keys", []),
        "sample_usage": meta.get("sample_usage")
    }


def format_metadata(table: Dict[str, Any]) -> str:
    lines = [f"# Table: {table['table_name']}", "["]
    if table.get("description"):
        lines.append(f"  Description: {table['description']}")

    for col in table["columns"]:
        desc = col.get("desc", col["name"])
        ex_str = f" Value examples: {[repr(e) for e in col.get('examples', [])[:5]]}" if col.get("examples") else ""
        range_str = f" (Range: {col['min']} ~ {col['max']})" if col["min"] or col["max"] else ""
        lines.append(f"  ({col['name']}, {desc}.{ex_str}{range_str}),")

    if lines[-1].endswith(","):
        lines[-1] = lines[-1][:-1]
    lines.append("]")
    return "\n".join(lines)


def get_schema_summary(db_id, table_names: List[str]) -> tuple[str, str, List[Dict[str, Any]]]:
    table_names = redis_client.smembers(f"{db_id}:metadata:table_names")

    schema_tables = []
    fk_lines = []

    for table in table_names:
        meta_raw = redis_client.get(f"{db_id}:metadata:{table}")
        meta = json.loads(meta_raw)
        schema = extract_metadata(table, meta)
        schema_tables.append(schema)

        for fk in meta.get("schema", {}).get("foreign_keys", []):
            from_col, to_table, to_col = fk
            fk_lines.append(f'{table}."{from_col}" = {to_table}."{to_col}"')

    schema_str = "\n\n".join(format_metadata(t) for t in schema_tables)
    fk_str = "\n".join(fk_lines)
    return schema_str, fk_str, schema_tables


def prune_schema_with_llm(db_id: str, schema_info: str, fk_info: str, query: str, evidence: str, llm: BaseChatModel):
    prompt = selector_template.format(
        db_id=db_id,
        desc_str=schema_info,
        fk_str=fk_info,
        query=query,
        evidence=evidence
    )
    res = call_llm(prompt, llm=llm)
    try:
        m = re.search(r"```json\s*([\s\S]+?)```", res)
        if not m:
            m = re.search(r"```([\s\S]+?)```", res)
        if m:
            res = m.group(1).strip()
        return json.loads(res)
    except Exception:
        print("❌ Failed to parse pruned schema:", res)
        return {}


def apply_pruning(schema_tables: List[Dict[str, Any]], extracted: Dict[str, Any]) -> List[Dict[str, Any]]:
    pruned = []
    for table in schema_tables:
        name = table["table_name"]
        decision = extracted.get(name, "drop_all")

        if decision == "drop_all":
            continue
        elif decision in ("keep_all", ""):
            pruned.append(table)
        elif isinstance(decision, list):
            table_copy = dict(table)
            table_copy["columns"] = [col for col in table["columns"] if col["name"] in decision]
            pruned.append(table_copy)

    return pruned


def selector_node(state, llm: BaseChatModel):
    db_id = state["db_id"]
    query = state["query"]
    evidence = state.get("evidence", "")

    vector_store = VectorStore(db_id=db_id)
    result = vector_store.query(query, k=15)

    table_names = [doc["table"] for doc in result]
    table_names = [t.decode() if isinstance(t, bytes) else t for t in table_names]

    schema_info, fk_info, schema_tables = get_schema_summary(db_id, table_names)
    too_large = schema_info.count("# Table:") > 6 or schema_info.count("(") > 30

    if too_large:
        print("📦 Schema too large. Pruning with LLM...")
        extracted = prune_schema_with_llm(db_id, schema_info, fk_info, query, evidence, llm)
        pruned_tables = apply_pruning(schema_tables, extracted)
        schema_info = "\n\n".join(format_metadata(t) for t in pruned_tables)
    else:
        extracted = {t["table_name"]: "keep_all" for t in schema_tables}

    return {
        **state,
        "desc_str": schema_info,
        "fk_str": fk_info,
        "extracted_schema": extracted,
        "pruned": too_large,
        "send_to": "decomposer_node",
        'messages': state.get("messages", []) + [
            {
                "role": "selector",
                "content": extracted
            }]
    }