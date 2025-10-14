from typing import TypedDict, Annotated, List, Optional, Dict, Any


class PreprocessState(TypedDict):
    # Inputs
    db_id: str
    final_sql: Annotated[Optional[str], None]
    df_raw: Annotated[Optional[object], None]  # pandas.DataFrame at runtime
    steps: Annotated[Optional[List[str]], None]

    # Options
    impute_strategy: Annotated[Optional[str], None]
    scaling: Annotated[Optional[str], None]
    one_hot_threshold: Annotated[Optional[int], None]
    fetch_only: Annotated[Optional[bool], None]
    persist_to_redis: Annotated[Optional[bool], None]

    # Outputs
    df_preprocessed: Annotated[Optional[object], None]
    preprocess_report: Annotated[Optional[Dict[str, Any]], None]
    column_stats: Annotated[Optional[Dict[str, Any]], None]
    feature_map: Annotated[Optional[Dict[str, Any]], None]
    warnings: Annotated[Optional[List[str]], None]
    df_redis_key: Annotated[Optional[str], None]
    df_shape: Annotated[Optional[tuple], None]
    columns: Annotated[Optional[List[str]], None]


