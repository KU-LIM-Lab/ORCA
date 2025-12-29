from typing import TypedDict, Annotated

class TableState(TypedDict):
    db_id: str 
    input: str
    table_analysis: Annotated[dict, None]
    related_tables: Annotated[dict, None]
    recommended_analysis: Annotated[str, None]
    final_output: Annotated[str, None]