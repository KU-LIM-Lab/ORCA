from typing import Optional
from .redis_client import redis_client
import pandas as pd
import io


def save_df_parquet(key: str, df: pd.DataFrame) -> None:
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    redis_client.set(key, buf.getvalue())


def load_df_parquet(key: str) -> Optional[pd.DataFrame]:
    data = redis_client.get(key)
    if not data:
        return None
    return pd.read_parquet(io.BytesIO(data))


