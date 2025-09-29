from typing import Optional, List, TypedDict, Annotated

class AgentState(TypedDict):
    messages: List # 
    db_id: str 
    query: str
    evidence: Optional[str] # if needed, related to the query

    desc_str: Optional[str]
    fk_str: Optional[str]
    extracted_schema: Optional[dict]
    pruned: bool
    pred: Optional[str]
    final_sql: Optional[str]
    qa_pairs: Optional[str]

    columns: Optional[List[str]]  
    result: Optional[List]
    error: Optional[str]
    send_to: str
    try_times: int
    llm_review: Optional[str]  
    output: Optional[dict]  