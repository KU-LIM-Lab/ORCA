
import re
from utils.llm import call_llm
from langchain_core.language_models.chat_models import BaseChatModel
from prompts.table_recommender_prompts import table_rec_prompt as prompt, table_rec_parser as parser

from utils.vectordb import VectorStore

def recommend_tables(state, llm: BaseChatModel):
    objective = state["objective_summary"]
    db_id = state.get("db_id", "daa")

    vector_store = VectorStore(db_id=db_id)
    results = vector_store.query(objective, k=15)
    lines = []
    for doc in results:
        table = doc["table"]
        text = doc["text"]
        columns = re.findall(r"- `(\w+)`:", text)
        lines.append(f"{table}: Columns: {', '.join(columns)}")
    table_list = "\n".join(lines)

    try:
        response = call_llm(
            prompt=prompt,
            parser=parser,
            variables={
                "objective": objective,
                "tables": table_list,
                "format_instructions": parser.get_format_instructions()
            }, llm=llm
        )
        state["recommended_tables"] = response.recommended_tables
        state["recommended_method"] = response.analysis_method
    except Exception as e:
        print(f"Error: Failed to parse recommended tables: {e}")
        state["recommended_tables"] = []
        state["recommended_method"] = "No specific method recommended."
    return state