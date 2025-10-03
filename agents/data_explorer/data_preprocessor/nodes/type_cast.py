from typing import Dict
import pandas as pd
import re


def type_cast_node(state: Dict) -> Dict:
    df = state.get("df_raw")
    if df is None:
        return state
    try:
        # More intelligent datetime detection
        for col in df.columns:
            if df[col].dtype == object:
                sample = df[col].dropna().astype(str).head(5).tolist()
                
                # Skip obvious non-date columns
                if any(keyword in col.lower() for keyword in ['id', 'password', 'hash', 'token', 'key', 'uuid']):
                    print(f"[TYPE_CAST] Skipping column '{col}' - appears to be identifier/hash")
                    continue
                
                # More sophisticated date pattern detection
                date_patterns = [
                    r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
                    r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
                    r'\d{4}/\d{2}/\d{2}',  # YYYY/MM/DD
                    r'\d{2}-\d{2}-\d{4}',  # MM-DD-YYYY
                ]
                
                # Check if sample contains date-like patterns
                has_date_pattern = False
                for pattern in date_patterns:
                    if any(re.match(pattern, s) for s in sample):
                        has_date_pattern = True
                        break
                
                # Also check for common date keywords
                date_keywords = ['date', 'time', 'created', 'updated', 'timestamp']
                has_date_keyword = any(keyword in col.lower() for keyword in date_keywords)
                
                if has_date_pattern or has_date_keyword:
                    try:
                        df[col] = pd.to_datetime(df[col])
                        print(f"[TYPE_CAST] Successfully converted column '{col}' to datetime")
                    except (ValueError, TypeError) as e:
                        print(f"[TYPE_CAST] Could not convert column '{col}' to datetime: {e}")
                        # Keep original column as-is
                        pass
                else:
                    print(f"[TYPE_CAST] Skipping column '{col}' - no date patterns detected")
        
        state["df_raw"] = df
        state["_done_type_cast"] = True
    except Exception as e:
        state.setdefault("warnings", []).append(f"type_cast failed: {e}")
    return state


