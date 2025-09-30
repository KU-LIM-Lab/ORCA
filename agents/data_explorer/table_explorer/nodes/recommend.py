from utils.llm import call_llm
from prompts.table_explorer_prompts import usecase_parser as parser, usecase_prompt as prompt
from langchain_core.language_models.chat_models import BaseChatModel
from .related import related_tables

def recommend_analysis(table_name: str, db_id: str, table_description: str, llm: BaseChatModel) -> str:
    table_name = table_name
    db_id = db_id
    related = related_tables(table_name, db_id)

    recommendation = call_llm(
        prompt=prompt,
        parser=parser,
        variables={
            "table_description": table_description,
            "related_tables": related
        },
        llm=llm
    )
    return recommendation

def recommend_analysis_node(state, llm):
    table_name = state["input"]
    db_id = state["db_id"]    
    full_description = state["table_analysis"]
    related_tables_info = state["related_tables"]

    columns_info = full_description.get("columns", [])
    table_description = full_description.get("table_description", "")

    recommendation = recommend_analysis(table_name, db_id, table_description, llm)

    final_output = {
        "table_name": state.get("input", ""),
        "table_analysis": {
            "table_description": table_description.strip(),
            "columns": [
                {
                    "column_name": col["column_name"],
                    "data_type": col["data_type"],
                    "nullable": col["nullable"],
                    "nulls": col["nulls"],
                    "notes": col.get("notes", [])
                }
                for col in columns_info
            ],
            "analysis_considerations": full_description.get("analysis_considerations", "")
        },
        "related_tables": {
            table: reason for table, reason in related_tables_info.items()
        },
        "recommended_analysis": [
            {
                "Analysis_Topic": uc.analysis_topic,
                "Suggested_Methodology": uc.suggested_methodology,
                "Expected_Insights": uc.expected_insights
            }
            for uc in recommendation.usecases
        ]
    }

    return {
        "recommended_analysis": recommendation,
        "final_output": final_output
    }