import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # ORCA/ root
sys.path.insert(0, str(Path(__file__).resolve().parent))      # data_exploration/

import json, os, argparse
from datetime import datetime

from langchain_core.prompts import ChatPromptTemplate

from baseline_prompts import (
    TABLE_EXPLORER_SYSTEM, TABLE_EXPLORER_HUMAN,
    TABLE_RECOMMENDER_SYSTEM, TABLE_RECOMMENDER_HUMAN,
    TEXT2SQL_SYSTEM, TEXT2SQL_HUMAN,
)
from utils.database import Database
from utils.llm import get_llm, call_llm
from utils.data_prep.metadata import generate_table_markdown
from utils.data_prep.related_tables import extract_schema


# ----- PROMPTS -----
table_explorer_prompt = ChatPromptTemplate.from_messages([
    ("system", TABLE_EXPLORER_SYSTEM),
    ("human", TABLE_EXPLORER_HUMAN),
])

table_recommender_prompt = ChatPromptTemplate.from_messages([
    ("system", TABLE_RECOMMENDER_SYSTEM),
    ("human", TABLE_RECOMMENDER_HUMAN),
])

text2sql_prompt = ChatPromptTemplate.from_messages([
    ("system", TEXT2SQL_SYSTEM),
    ("human", TEXT2SQL_HUMAN),
])


# ----- SCHEMA CACHE -----
_schema_cache: dict = {}


def get_schema_info(db_id: str) -> dict:
    """Fetch and cache full schema with statistics for a database."""
    if db_id not in _schema_cache:
        print(f"Fetching schema for '{db_id}'... (this may take a moment)")
        _schema_cache[db_id] = extract_schema(db_id)
        print(f"Schema cached: {len(_schema_cache[db_id])} tables found.")
    return _schema_cache[db_id]


# ----- HELPERS -----
def fetch_sample_rows(db_id: str, table_name: str, limit: int = 5) -> str:
    """Fetch sample rows and format as a markdown table."""
    db = Database()
    try:
        rows, cols = db.run_query(f'SELECT * FROM "{table_name}" LIMIT {limit}', db_id)
        if not rows:
            return "(no rows found)"
        header = " | ".join(str(c) for c in cols)
        separator = " | ".join("---" for _ in cols)
        body = "\n".join(
            " | ".join(str(v) if v is not None else "NULL" for v in row)
            for row in rows
        )
        return f"{header}\n{separator}\n{body}"
    except Exception as e:
        return f"(error fetching rows: {e})"


def format_table_list(schema_info: dict) -> str:
    """Format schema info as a compact table list for the recommender prompt."""
    lines = []
    for table, info in schema_info.items():
        col_parts = []
        for col, meta in info["columns"].items():
            col_desc = col
            if meta.get("pk"):
                col_desc += " [PK]"
            elif meta.get("fk"):
                col_desc += f" [FK→{meta['fk']}]"
            col_parts.append(col_desc)
        lines.append(f"- {table}: {', '.join(col_parts)}")
    return "\n".join(lines)


def format_schema_as_ddl(schema_info: dict) -> str:
    """Format schema info as CREATE TABLE DDL statements."""
    parts = []
    for table, info in schema_info.items():
        col_defs = []
        for col, meta in info["columns"].items():
            col_def = f"  {col} {meta['type']}"
            if meta.get("pk"):
                col_def += " PRIMARY KEY"
            if meta.get("nullable") is False:
                col_def += " NOT NULL"
            col_defs.append(col_def)
        for fk_col, ref_table, ref_col in info.get("foreign_keys", []):
            col_defs.append(
                f"  FOREIGN KEY ({fk_col}) REFERENCES {ref_table}({ref_col})"
            )
        parts.append(f"CREATE TABLE {table} (\n" + ",\n".join(col_defs) + "\n);")
    return "\n\n".join(parts)


def parse_json_list(text: str) -> list:
    """Parse a JSON array from LLM response, with fallback."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
    except (json.JSONDecodeError, ValueError):
        pass
    return []


def strip_sql_markdown(text: str) -> str:
    """Strip markdown code fences from SQL output."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        inner = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
        text = "\n".join(inner).strip()
    return text


# ----- JSON HELPERS -----
class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        try:
            return str(obj)
        except Exception:
            return super().default(obj)


def load_questions(path: Path) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_results(path: Path, results: list):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, cls=EnhancedJSONEncoder)
    jsonl_path = path.with_suffix(".jsonl")
    if jsonl_path.exists():
        jsonl_path.unlink()


def save_checkpoint(result: dict, result_dir: Path, prefix: str, data: str, timestamp: str):
    file_path = result_dir / f"{prefix}_{data}_{timestamp}.jsonl"
    with open(file_path, "a", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, cls=EnhancedJSONEncoder)
        f.write("\n")


# ----- EXPERIMENT FUNCTIONS -----
def experiment_table_explorer(
    data_dir: Path, result_dir: Path, timestamp: str, data: str, llm, db_id: str
):
    questions = load_questions(data_dir / "table_description.json")
    schema_info = get_schema_info(db_id)
    results = []

    for i, q in enumerate(questions):
        table_name = q["table_name"]
        print(f"[{i+1}/{len(questions)}] Table: {table_name}")
        try:
            table_schema = {table_name: schema_info[table_name]}
            schema_md = generate_table_markdown(table_schema)
            sample_rows = fetch_sample_rows(db_id, table_name, limit=100)
            description = call_llm(
                prompt=table_explorer_prompt,
                variables={
                    "table_name": table_name,
                    "schema_markdown": schema_md,
                    "sample_rows": sample_rows,
                    "sample_limit": 100,
                },
                llm=llm,
            )
            result_entry = {
                "table_name": table_name,
                "db_id": db_id,
                "description": description,
            }
        except Exception as e:
            print(f"  Error: {e}")
            result_entry = {
                "table_name": table_name,
                "db_id": db_id,
                "description": None,
                "error": str(e),
            }

        results.append(result_entry)
        save_checkpoint(result_entry, result_dir, "baseline_table_explorer", data, timestamp)

    out_path = result_dir / f"baseline_table_explorer_{data}_{timestamp}.json"
    save_results(out_path, results)
    print(f"\nTable Explorer done. → {out_path}")


def experiment_table_recommender(
    data_dir: Path, result_dir: Path, timestamp: str, data: str, llm, db_id: str
):
    questions = load_questions(data_dir / "table_recommendation.json")
    schema_info = get_schema_info(db_id)
    table_list_str = format_table_list(schema_info)
    results = []

    for i, q in enumerate(questions):
        print(f"[{i+1}/{len(questions)}] Q: {q['question'][:60]}...")
        try:
            raw = call_llm(
                prompt=table_recommender_prompt,
                variables={
                    "question": q["question"],
                    "table_list": table_list_str,
                },
                llm=llm,
            )
            predicted = parse_json_list(raw)
            result_entry = {
                "question": q["question"],
                "db_id": db_id,
                "ground_truth": q["answer"],
                "predicted_tables": predicted,
                "raw_response": raw,
            }
        except Exception as e:
            print(f"  Error: {e}")
            result_entry = {
                "question": q["question"],
                "db_id": db_id,
                "ground_truth": q["answer"],
                "predicted_tables": [],
                "raw_response": None,
                "error": str(e),
            }

        results.append(result_entry)
        save_checkpoint(result_entry, result_dir, "baseline_table_recommender", data, timestamp)

    out_path = result_dir / f"baseline_table_recommender_{data}_{timestamp}.json"
    save_results(out_path, results)
    print(f"\nTable Recommender done. → {out_path}")


def experiment_text2sql(
    data_dir: Path, result_dir: Path, timestamp: str, data: str, llm, db_id: str
):
    questions = load_questions(data_dir / "text2sql.json")
    schema_info = get_schema_info(db_id)
    schema_ddl = format_schema_as_ddl(schema_info)
    results = []

    for i, q in enumerate(questions):
        print(f"[{i+1}/{len(questions)}] Q{q['question_id']}: {q['question'][:60]}...")
        try:
            evidence = q.get("evidence", "").strip()
            evidence_block = f"\n-- External Knowledge: {evidence}" if evidence else ""
            raw = call_llm(
                prompt=text2sql_prompt,
                variables={
                    "question": q["question"],
                    "evidence_block": evidence_block,
                    "schema_ddl": schema_ddl,
                },
                llm=llm,
            )
            generated_sql = strip_sql_markdown(raw)
            result_entry = {
                "question_id": q["question_id"],
                "question": q["question"],
                "db_id": db_id,
                "ground_truth_sql": q["SQL"],
                "generated_sql": generated_sql,
                "difficulty": q.get("difficulty", ""),
                "error": None,
            }
        except Exception as e:
            print(f"  Error: {e}")
            result_entry = {
                "question_id": q["question_id"],
                "question": q["question"],
                "db_id": db_id,
                "ground_truth_sql": q["SQL"],
                "generated_sql": None,
                "difficulty": q.get("difficulty", ""),
                "error": str(e),
            }

        results.append(result_entry)
        save_checkpoint(result_entry, result_dir, "baseline_text2sql", data, timestamp)

    out_path = result_dir / f"baseline_text2sql_{data}_{timestamp}.json"
    save_results(out_path, results)
    print(f"\nText2SQL done. → {out_path}")


# ----- MAIN -----
def main():
    parser = argparse.ArgumentParser(
        description="Run baseline data exploration experiments (direct LLM, no agents)"
    )
    parser.add_argument(
        "-d", "--data", type=str, choices=["daa", "bird"], default="daa",
        help="Dataset name"
    )
    parser.add_argument(
        "-t", "--task", type=str, nargs="+",
        choices=["table_explorer", "table_recommender", "text2sql"],
        default=["table_explorer", "table_recommender", "text2sql"],
        help="Tasks to run"
    )
    parser.add_argument("-m", "--model", type=str, default="gpt-4o-mini")
    parser.add_argument(
        "-p", "--provider", type=str, default="openai",
        choices=["openai", "google", "ollama"]
    )
    parser.add_argument(
        "--db-id", type=str, default="reef_db",
        help="Database ID to query (default: reef_db)"
    )
    args = parser.parse_args()

    llm = get_llm(model=args.model, temperature=0.3, provider=args.provider)
    model_tag = f"{args.provider}_{args.model.replace('/', '-')}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    data_dir = Path(f"experiments_v0/questions/{args.data}")
    result_dir = Path(f"experiments_v0/results/{args.data}/{model_tag}")
    os.makedirs(result_dir, exist_ok=True)

    print(f"Model: {args.model}  Provider: {args.provider}  Dataset: {args.data}  DB: {args.db_id}")
    print(f"Results → {result_dir}\n")

    tasks = set(args.task)
    if "table_explorer" in tasks:
        experiment_table_explorer(data_dir, result_dir, timestamp, args.data, llm, args.db_id)
    if "table_recommender" in tasks:
        experiment_table_recommender(data_dir, result_dir, timestamp, args.data, llm, args.db_id)
    if "text2sql" in tasks:
        experiment_text2sql(data_dir, result_dir, timestamp, args.data, llm, args.db_id)


if __name__ == "__main__":
    main()
