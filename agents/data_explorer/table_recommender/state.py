from typing import Annotated, TypedDict, Literal

class RecommendState(TypedDict):
    db_id: str
    input_type: Literal["text", "document"]
    input: str
    parsed_text: Annotated[str, None]
    objective_summary: Annotated[str, None]
    recommended_tables: Annotated[list, None]
    recommended_method: Annotated[str, None]
    erd_image_path: Annotated[str, None]
    final_output: Annotated[str, None]