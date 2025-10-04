# causal_analysis/nodes/parse_question.py
import json
from typing import Callable, Dict

from utils.llm import call_llm
from utils.data_prep.metadata import generate_table_markdown
from utils.query_validation import extract_table_metadata, validate_parsed_query, sanitize_parsed_query
from prompts.causal_analysis_prompts import parse_query_prompt, parse_query_parser

from langchain_core.runnables import RunnableLambda
from langchain_core.language_models.chat_models import BaseChatModel

from utils.redis_client import redis_client

def build_parse_question_node(llm: BaseChatModel) -> Callable:
    def _parse_question(state: Dict) -> Dict:
        if state.get("variable_info"): # variable_info is already provided
            state["parsed_query"] = state["variable_info"]
            return state

        question = state["input"] # natural language question is provided
        db_id = state["db_id"]

        table_keys = redis_client.keys(f"{db_id}:metadata:*")
        table_markdowns = []

        for key in table_keys:
            if key == f"{db_id}:metadata:table_names":
                continue
            table_name = key.split(":")[2]
            raw = redis_client.get(key)
            if not raw:
                continue
            try:
                metadata = json.loads(raw)
                schema = metadata.get("schema", {})
                markdown = generate_table_markdown({table_name: schema})
                table_markdowns.append(markdown)
            except Exception as e:
                print(f"⚠️ Error parsing metadata for key {key}: {e}")

        full_markdown = "\n\n".join(table_markdowns)
        table_metadata, primary_keys = extract_table_metadata(table_markdowns)

        parsed = None
        issues = []

        for attempt in range(3):
            fix_note = ""
            if attempt > 0 and issues:
                fix_note = "\n\nFix Note:\n" + "\n".join(issues)

            result = call_llm(
                prompt=parse_query_prompt,
                parser=parse_query_parser,
                variables={
                    "question": question,
                    "tables": full_markdown + fix_note,
                    "expression_dict": json.dumps(state["expression_dict"]) 
                },
                llm=llm
            )

            parsed = result.dict()
            issues = validate_parsed_query(parsed, table_metadata, primary_keys)
            if not issues:
                break

        parsed = sanitize_parsed_query(parsed, table_metadata, primary_keys)
        state["parsed_query"] = parsed
        state["table_schema_str"] = table_markdowns
        return state

    return RunnableLambda(_parse_question)