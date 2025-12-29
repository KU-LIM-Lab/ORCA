"""
ATE 계산 모듈
DoWhy를 사용하여 변수 타입에 따라 적절한 estimator를 선택하여 ATE를 계산합니다.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dowhy import CausalModel
import statsmodels.api as sm
import warnings

warnings.filterwarnings('ignore')


def detect_variable_type(series: pd.Series) -> str:
    """
    변수의 타입을 감지합니다.
    
    Returns:
        'binary': 0/1 또는 True/False만 포함
        'continuous': 연속형 숫자
        'count': 정수형 카운트 변수
        'categorical': 범주형
    """
    # 결측값 제거
    series_clean = series.dropna()
    
    if len(series_clean) == 0:
        return 'continuous'  # 기본값
    
    # Binary 체크
    unique_vals = series_clean.unique()
    if len(unique_vals) == 2:
        if set(unique_vals).issubset({0, 1, True, False, 0.0, 1.0}):
            return 'binary'
    
    # 정수형이고 범위가 작으면 count로 간주
    if pd.api.types.is_integer_dtype(series_clean):
        if series_clean.min() >= 0 and series_clean.max() < 1000:
            return 'count'
    
    # 범주형 체크 (문자열이거나 unique 값이 적은 경우)
    if pd.api.types.is_object_dtype(series_clean) or pd.api.types.is_categorical_dtype(series_clean):
        if len(unique_vals) <= 20:  # 범주가 20개 이하면 범주형으로 간주
            return 'categorical'
    
    # 기본적으로 연속형
    return 'continuous'


def select_estimator(treatment_type: str, outcome_type: str) -> str:
    """
    변수 타입에 따라 적절한 estimator를 선택합니다.
    
    Args:
        treatment_type: treatment 변수의 타입
        outcome_type: outcome 변수의 타입
    
    Returns:
        estimator 이름 (예: 'backdoor.linear_regression')
    """
    # Binary outcome인 경우
    if outcome_type == 'binary':
        return 'backdoor.generalized_linear_model'
    
    # Count outcome인 경우 (정수형)
    if outcome_type == 'count':
        return 'backdoor.generalized_linear_model'
    
    # Continuous outcome인 경우
    if outcome_type == 'continuous':
        return 'backdoor.linear_regression'
    
    # 기본값
    return 'backdoor.linear_regression'


def coerce_df_to_numeric(
    df: pd.DataFrame,
    dropna: bool = True,
    datetime_unit: str = "s",   # "s" or "ms"
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Convert a DataFrame into a fully numeric DataFrame
    suitable for statsmodels / DoWhy / sklearn.

    Rules:
    - bool        -> int (0/1)
    - category    -> codes
    - object      -> try numeric, otherwise categorical codes
    - datetime    -> unix timestamp
    - remaining   -> pd.to_numeric(coerce)

    Parameters
    ----------
    dropna : bool
        Whether to drop rows with NaN after conversion
    datetime_unit : {"s", "ms"}
        Unit for datetime conversion
    """

    df_out = df.copy()

    for col in df_out.columns:
        s = df_out[col]

        # 1) Boolean → int
        if pd.api.types.is_bool_dtype(s):
            df_out[col] = s.astype(int)
            if verbose:
                print(f"[bool -> int] {col}")
            continue

        # 2) Datetime → timestamp
        if pd.api.types.is_datetime64_any_dtype(s):
            factor = 1e9 if datetime_unit == "s" else 1e6
            df_out[col] = s.astype("int64") / factor
            if verbose:
                print(f"[datetime -> ts] {col}")
            continue

        # 3) Categorical → codes
        if pd.api.types.is_categorical_dtype(s):
            df_out[col] = s.cat.codes
            if verbose:
                print(f"[category -> codes] {col}")
            continue

        # 4) Object → try numeric → fallback to categorical
        if pd.api.types.is_object_dtype(s):
            # strip common noise
            s_clean = (
                s.astype(str)
                 .str.replace(",", "", regex=False)
                 .str.replace("₩", "", regex=False)
                 .str.strip()
            )
            numeric = pd.to_numeric(s_clean, errors="coerce")

            # if "mostly numeric", treat as numeric
            if numeric.notna().mean() > 0.8:
                df_out[col] = numeric
                if verbose:
                    print(f"[object -> numeric] {col}")
            else:
                df_out[col] = pd.Categorical(s).codes
                if verbose:
                    print(f"[object -> categorical codes] {col}")
            continue

        # 5) Everything else → force numeric
        df_out[col] = pd.to_numeric(s, errors="coerce")
        if verbose:
            print(f"[forced numeric] {col}")

    if dropna:
        before = len(df_out)
        df_out = df_out.dropna()
        if verbose:
            print(f"Dropped {before - len(df_out)} rows due to NaN")

    return df_out

def create_causal_graph_from_data(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    confounders: List[str],
    mediators: List[str] = None,
    instrumental_variables: List[str] = None
) -> str:
    """
    데이터에서 간단한 causal graph를 생성합니다.
    confounders -> treatment, outcome
    treatment -> outcome
    mediators가 있으면: treatment -> mediators -> outcome
    """
    if mediators is None:
        mediators = []
    if instrumental_variables is None:
        instrumental_variables = []
    
    edges = []
    
    # Confounders -> treatment, outcome
    for conf in confounders:
        if conf in df.columns:
            edges.append(f'"{conf}" -> "{treatment}"')
            edges.append(f'"{conf}" -> "{outcome}"')
    
    # Treatment -> outcome
    edges.append(f'"{treatment}" -> "{outcome}"')
    
    # Mediators: treatment -> mediator -> outcome
    for med in mediators:
        if med in df.columns:
            edges.append(f'"{treatment}" -> "{med}"')
            edges.append(f'"{med}" -> "{outcome}"')
    
    # Instrumental variables: IV -> treatment (outcome과 직접 연결 없음)
    for iv in instrumental_variables:
        if iv in df.columns:
            edges.append(f'"{iv}" -> "{treatment}"')
    
    dot_graph = "digraph {\n" + "\n".join(edges) + "\n}"
    return dot_graph


def calculate_ate(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    confounders: List[str] = None,
    mediators: List[str] = None,
    instrumental_variables: List[str] = None,
    estimator: Optional[str] = None,
    causal_graph: Optional[str] = None
) -> Dict[str, Any]:
    """
    DoWhy를 사용하여 ATE를 계산합니다.
    
    Args:
        df: 데이터프레임
        treatment: treatment 변수명
        outcome: outcome 변수명
        confounders: 교란 변수 리스트
        mediators: 매개 변수 리스트
        instrumental_variables: 도구 변수 리스트
        estimator: 사용할 estimator (None이면 자동 선택)
        causal_graph: DAG 그래프 (DOT 형식, None이면 자동 생성)
    
    Returns:
        ATE 계산 결과 딕셔너리
    """
    if confounders is None:
        confounders = []
    if mediators is None:
        mediators = []
    if instrumental_variables is None:
        instrumental_variables = []
    
    # 변수 존재 확인
    if treatment not in df.columns:
        raise ValueError(f"Treatment variable '{treatment}' not found in dataframe")
    if outcome not in df.columns:
        raise ValueError(f"Outcome variable '{outcome}' not found in dataframe")
    
    # 결측값이 있는 행 제거
    required_cols = [treatment, outcome] + confounders + mediators + instrumental_variables
    required_cols = [col for col in required_cols if col in df.columns]
    df_clean = df[required_cols].dropna()
    df_clean = coerce_df_to_numeric(df_clean)
    
    if len(df_clean) < 10:
        raise ValueError(f"Insufficient data: {len(df_clean)} rows after cleaning, need at least 10")
    
    # 변수 타입 감지
    treatment_type = detect_variable_type(df_clean[treatment])
    outcome_type = detect_variable_type(df_clean[outcome])
    
    # Estimator 선택
    if estimator is None:
        estimator = select_estimator(treatment_type, outcome_type)
    
    # Causal graph 생성
    if causal_graph is None:
        causal_graph = create_causal_graph_from_data(
            df_clean, treatment, outcome, confounders, mediators, instrumental_variables
        )
    
    # CausalModel 생성
    try:
        model = CausalModel(
            data=df_clean,
            treatment=treatment,
            outcome=outcome,
            common_causes=confounders if confounders else None,
            instruments=instrumental_variables if instrumental_variables else None,
            mediators=mediators if mediators else None,
            graph=causal_graph
        )
    except Exception as e:
        raise RuntimeError(f"Failed to construct CausalModel: {e}")
    
    # Identification
    try:
        identified_estimand = model.identify_effect()
    except Exception as e:
        raise RuntimeError(f"Failed to identify effect: {e}")
    
    # Estimation method parameters
    method_params = {}
    
    # GLM 설정
    if estimator == "backdoor.generalized_linear_model":
        y = df_clean[outcome].dropna()
        is_binary = (
            y.nunique() == 2 and
            set(y.unique()).issubset({0, 1, True, False, 0.0, 1.0})
        )
        
        if is_binary:
            method_params["glm_family"] = sm.families.Binomial()
        else:
            # Count 데이터인 경우 Poisson 또는 Negative Binomial
            if outcome_type == 'count':
                method_params["glm_family"] = sm.families.Poisson()
            else:
                method_params["glm_family"] = sm.families.Gaussian()
    
    # Propensity score matching 등에서 사용할 분류 모델
    classification_estimators = [
        "backdoor.propensity_score_matching",
        "backdoor.propensity_score_stratification",
        "backdoor.propensity_score_weighting",
        "backdoor.distance_matching"
    ]
    
    if estimator in classification_estimators:
        try:
            from tabpfn import TabPFNClassifier
            method_params["propensity_score_model"] = TabPFNClassifier()
        except ImportError:
            # TabPFN이 없으면 기본 분류기 사용
            from sklearn.linear_model import LogisticRegression
            method_params["propensity_score_model"] = LogisticRegression(max_iter=1000)
        
        # 범주형 변수 인코딩
        cat_cols = df_clean.select_dtypes(include=["category", "object"]).columns
        if len(cat_cols) > 0:
            for col in cat_cols:
                if col in required_cols:
                    df_clean[col] = pd.Categorical(df_clean[col]).codes
    
    # Estimate causal effect
    try:
        estimate = model.estimate_effect(
            identified_estimand,
            method_name=estimator,
            method_params=method_params if method_params else None
        )
    except Exception as e:
        raise RuntimeError(f"Failed to estimate causal effect: {e}")
    
    # 결과 추출
    try:
        ate_value = float(getattr(estimate, "value", None)) if getattr(estimate, "value", None) is not None else None
    except (ValueError, TypeError):
        ate_value = None
    
    try:
        ci_raw = estimate.get_confidence_intervals()
        ci_arr = np.asarray(ci_raw, dtype=float)
        
        if ci_arr.ndim == 2 and ci_arr.shape[0] == 1:
            ci_arr = ci_arr[0]
        
        confidence_interval = ci_arr.tolist() if ci_arr.size > 0 else None
    except Exception:
        confidence_interval = None
    
    try:
        p_value = getattr(estimate, "p_value", None)
    except Exception:
        p_value = None
    
    return {
        "ate": ate_value,
        "confidence_interval": confidence_interval,
        "p_value": p_value,
        "estimator": estimator,
        "treatment_type": treatment_type,
        "outcome_type": outcome_type,
        "n_samples": len(df_clean),
        "model": model,
        "estimate": estimate
    }

