from typing import TypedDict, Annotated, List, Optional, Dict, Any


class PreprocessState(TypedDict):
    # Inputs
    db_id: str
    final_sql: Annotated[Optional[str], None]
    df_raw: Annotated[Optional[object], None]  # pandas.DataFrame at runtime
    steps: Annotated[Optional[List[str]], None]

    # Options

    # Outputs
    df_redis_key: Annotated[Optional[str], None]
    df_shape: Annotated[Optional[tuple], None]


