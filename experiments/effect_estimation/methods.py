# experiments/effect_estimation/methods.py

from __future__ import annotations
from typing import Callable, Dict, Any
import numpy as np

# --- 공통 registry ---

METHOD_REGISTRY: Dict[str, Callable[[np.ndarray, np.ndarray, np.ndarray, dict | None], dict]] = {}


def register_method(name: str):
    """Decorator to register effect estimation methods by name."""
    def decorator(fn: Callable):
        METHOD_REGISTRY[name] = fn
        return fn
    return decorator


def get_method(name: str) -> Callable:
    if name not in METHOD_REGISTRY:
        raise ValueError(f"Unknown method: {name}. Available: {list(METHOD_REGISTRY.keys())}")
    return METHOD_REGISTRY[name]


# === 1. ORCA  ===

@register_method("orca")
def orca_agent_effect(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    context: dict | None = None,
) -> dict:
    """
    ORCA causal analysis agent를 통한 효과 추정.
    context:
        - "df": 전체 DataFrame (W, T, Y 포함)
        - "causal_graph": ORCA에서 사용하는 causal_graph dict
        - "app": generate_causal_analysis_graph로 만든 LangGraph app
        - "treatment_name": str
        - "outcome_name": str
    """
    if context is None:
        context = {}

    df = context["df"]
    causal_graph = context.get("causal_graph")
    if causal_graph is None:
        raise ValueError("orca method requires 'causal_graph' in context.")
    app = context["app"]
    treatment_name = context["treatment_name"]
    outcome_name = context["outcome_name"]

    question = f"What is the causal effect of {treatment_name} on {outcome_name}?"

    state_input = {
        "input": question,
        "df_preprocessed": df,
        "causal_graph": causal_graph,
    }

    result = app.invoke(state_input)

    tau_hat_ate = result.get("causal_effect_ate")
    ate_ci = result.get("causal_effect_ci")
    tau_hat_cate = result.get("cate_effects")  # 있으면 사용, 없으면 None

    return {
        "tau_hat_ate": float(tau_hat_ate) if tau_hat_ate is not None else None,
        "tau_hat_cate": np.asarray(tau_hat_cate) if tau_hat_cate is not None else None,
        "ate_ci": ate_ci,
        "raw": result,
    }


# === 2. Causal Agent (Han, 2024) ===

@register_method("causal_agent")
def causal_agent_effect(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    context: dict | None = None,
) -> dict:
    """
    Causal Agent를 통한 효과 추정.
    context:
        - "openai_api_key": OpenAI API key (optional, 환경변수에서도 읽을 수 있음)
        - "model_name": 모델 이름 (default: "gpt-3.5-turbo")
        - "base_url": API base URL (optional)
        - "feature_names": feature 이름 리스트 (optional)
        - "treatment_name": treatment 변수명 (default: "T")
        - "outcome_name": outcome 변수명 (default: "Y")
        - "T0": treatment 값 0 (default: 0.0)
        - "T1": treatment 값 1 (default: 1.0)
        - "agent_executor": AgentExecutor 객체 (optional, 재사용 가능)
    """
    if context is None:
        context = {}

    # Import from Causal_Agent subdirectory
    # Use relative import or add path manipulation
    try:
        from methods.Causal_Agent.agent_api import (
            build_causal_agent,
            causal_agent_effect_once,
        )
    except ImportError:
        # Fallback: add parent directory to path
        import sys
        from pathlib import Path
        methods_dir = Path(__file__).parent
        if str(methods_dir) not in sys.path:
            sys.path.insert(0, str(methods_dir))
        from methods.Causal_Agent.agent_api import (
            build_causal_agent,
            causal_agent_effect_once,
        )

    api_key = context.get("openai_api_key")
    model_name = context.get("model_name", "gpt-3.5-turbo")
    base_url = context.get("base_url")
    feature_names = context.get("feature_names")
    treatment_name = context.get("treatment_name", "T")
    outcome_name = context.get("outcome_name", "Y")
    T0 = context.get("T0", 0.0)
    T1 = context.get("T1", 1.0)
    agent_executor = context.get("agent_executor")

    result = causal_agent_effect_once(
        X=X,
        T=T,
        Y=Y,
        feature_names=feature_names,
        treatment_name=treatment_name,
        outcome_name=outcome_name,
        T0=T0,
        T1=T1,
        agent_executor=agent_executor,
    )
    return result


# # === 3. DoubleML ===

@register_method("doubleml")
def doubleml_effect(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    context: dict | None = None,
) -> dict:
    """
    Double Machine Learning 기반 ATE 추정 (ATE only).
    """
    from doubleml import DoubleMLData, DoubleMLPLR
    from sklearn.ensemble import RandomForestRegressor

    data = DoubleMLData.from_arrays(
        x=X,
        y=Y,
        d=T,
    )
    seed = context.get("seed", 0) if context else 0
    ml_m = RandomForestRegressor(random_state=seed) # treatment model
    ml_l = RandomForestRegressor(random_state=seed) # outcome model

    dml_plr = DoubleMLPLR(data, ml_l=ml_l, ml_m=ml_m, n_folds=3)
    dml_plr.fit()

    ate = float(dml_plr.coef)
    se = float(dml_plr.se)
    ci_lower = ate - 1.96 * se
    ci_upper = ate + 1.96 * se

    return {
        "tau_hat_ate": ate,
        "tau_hat_cate": None,  # 기본 DoubleMLPLR는 CATE는 직접 제공 X
        "ate_ci": (ci_lower, ci_upper),
        "raw": dml_plr,
    }


# === 4. Causal Forest (econml CausalForestDML) ===

@register_method("causal_forest")
def causal_forest_effect(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    context: dict | None = None,
) -> dict:
    """
    CausalForestDML 기반 ATE/CATE 추정.
    """
    from econml.dml import CausalForestDML
    from sklearn.ensemble import RandomForestRegressor

    est = CausalForestDML(
        model_t=RandomForestRegressor(),
        model_y=RandomForestRegressor(),
        n_estimators=500,
        min_samples_leaf=10,
        random_state=context.get("seed", 0) if context else 0,
    )
    est.fit(Y, T, X=X)

    tau_hat_cate = est.effect(X)
    tau_hat_ate = float(np.mean(tau_hat_cate))

    # CI (per-unit) → 평균으로 ATE CI 근사
    ci_lower, ci_upper = est.effect_interval(X)
    ate_ci = (float(ci_lower.mean()), float(ci_upper.mean()))

    return {
        "tau_hat_ate": tau_hat_ate,
        "tau_hat_cate": tau_hat_cate,
        "ate_ci": ate_ci,
        "raw": est,
    }


# === 5. T-Learner ===

@register_method("t_learner")
def t_learner_effect(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    context: dict | None = None,
) -> dict:
    """
    RandomForest 기반 T-Learner.
    """
    from sklearn.ensemble import RandomForestRegressor

    if context is None:
        context = {}
    seed = context.get("seed", 0)

    model_t0 = RandomForestRegressor(random_state=seed)
    model_t1 = RandomForestRegressor(random_state=seed)

    X0, Y0 = X[T == 0], Y[T == 0]
    X1, Y1 = X[T == 1], Y[T == 1]

    model_t0.fit(X0, Y0)
    model_t1.fit(X1, Y1)

    mu0_hat = model_t0.predict(X)
    mu1_hat = model_t1.predict(X)
    tau_hat_cate = mu1_hat - mu0_hat
    tau_hat_ate = float(tau_hat_cate.mean())

    # CI는 간단히 None, 필요하면 bootstrap으로 추가
    return {
        "tau_hat_ate": tau_hat_ate,
        "tau_hat_cate": tau_hat_cate,
        "ate_ci": None,
        "raw": {
            "model_t0": model_t0,
            "model_t1": model_t1,
        },
    }
