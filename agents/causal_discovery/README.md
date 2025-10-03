# CausalDiscoveryAgent 사용법

## 개요

CausalDiscoveryAgent는 인과관계 발견을 위한 전문 에이전트입니다. 5단계 파이프라인을 통해 데이터의 가정을 검증하고, 적절한 알고리즘을 선택하여 인과 그래프를 생성합니다.

## 파이프라인 구조

### 1. assumption_method_matrix

- **목적**: 각 변수쌍에 대해 가정을 테스트하고 점수 생성
- **출력**: `assumption_method_scores` (S_lin, S_nG, S_ANM, S_Gauss, S_EqVar)

### 2. algorithm_scoring

- **목적**: 가정 호환성에 기반하여 알고리즘 점수화
- **출력**: `algorithm_scores`, `selected_algorithms` (Top-k)

### 3. run_algorithms

- **목적**: 선택된 알고리즘들을 병렬로 실행
- **출력**: `algorithm_results` (각 알고리즘의 그래프와 메타데이터)

### 4. intermediate_scoring

- **목적**: 견고성, 충실도, 일반화 가능성 평가
- **출력**: `intermediate_scores`, `candidate_graphs`

### 5. final_graph_selection

- **목적**: 최종 인과 그래프 선택
- **출력**: `selected_graph`, `graph_selection_reasoning`

## 사용법

### 기본 사용법

```python
from agents.causal_discovery.agent import CausalDiscoveryAgent
from core.state import create_initial_state

# Agent 생성
agent = CausalDiscoveryAgent(
    name="causal_discovery",
    config={
        "bootstrap_iterations": 100,
        "cv_folds": 5,
        "top_k_algorithms": 3,
        "lambda_soft_and": 0.7,
        "beta_conservative": 2.0
    }
)

# 상태 생성 (전처리된 데이터 필요)
state = create_initial_state("분석 쿼리", "db_id")
state["df_preprocessed"] = your_dataframe  # pandas DataFrame
state["current_substep"] = "assumption_method_matrix"

# 단계별 실행
state = agent.step(state)
```

### 전체 파이프라인 실행

```python
# 1. 가정-방법 매트릭스 생성
state["current_substep"] = "assumption_method_matrix"
state = agent.step(state)

# 2. 알고리즘 점수화
state["current_substep"] = "algorithm_scoring"
state = agent.step(state)

# 3. 알고리즘 실행
state["current_substep"] = "run_algorithms"
state = agent.step(state)

# 4. 중간 점수 계산
state["current_substep"] = "intermediate_scoring"
state = agent.step(state)

# 5. 최종 그래프 선택
state["current_substep"] = "final_graph_selection"
state = agent.step(state)
```

## 설정 옵션

```python
config = {
    "bootstrap_iterations": 100,    # 부트스트랩 반복 횟수
    "cv_folds": 5,                  # 교차 검증 폴드 수
    "top_k_algorithms": 3,          # 선택할 상위 알고리즘 수
    "lambda_soft_and": 0.7,         # Soft-AND 가중치
    "beta_conservative": 2.0        # 보수적 변환 지수
}
```

## 도구 (Tools)

### StatsTool

- `linearity_test()`: GLM vs GAM 비교로 선형성 테스트
- `normality_test()`: Jarque-Bera, Shapiro-Wilk으로 정규성 테스트
- `gaussian_eqvar_test()`: 가우시안 및 등분산 가정 테스트

### IndependenceTool

- `anm_test()`: ANM 가정을 위한 독립성 테스트

### 알고리즘 도구

- `LiNGAMTool`: DirectLiNGAM 알고리즘
- `ANMTool`: Additive Noise Model 알고리즘
- `PCTool`: PC 알고리즘
- `GESTool`: GES 알고리즘
- `CAMTool`: CAM 알고리즘

### 평가 도구

- `Bootstrapper`: 부트스트랩 평가
- `GraphEvaluator`: BIC/MDL, 교차검증 평가
- `GraphOps`: 그래프 변환 및 병합

## 출력 데이터

### assumption_method_scores

```python
{
    "S_lin": {"var1_var2": 0.7, ...},      # 선형성 점수
    "S_nG": {"var1_var2": 0.6, ...},       # 비가우시안 점수
    "S_ANM": {"var1_var2": 0.8, ...},      # ANM 점수
    "S_Gauss": {"var1_var2": 0.9, ...},    # 가우시안 점수
    "S_EqVar": {"var1_var2": 0.7, ...}     # 등분산 점수
}
```

### algorithm_scores

```python
{
    "LiNGAM": {
        "final_score": 0.85,
        "weighted_score": 0.80,
        "soft_and_score": 0.90,
        "individual_scores": {...}
    },
    "ANM": {...},
    "PC": {...},
    "GES": {...},
    "CAM": {...}
}
```

### selected_graph

```python
# selected_graph는 최종 후보의 graph 딕셔너리입니다 (edges, variables 포함).
# metadata는 candidate_graphs의 각 항목에 포함됩니다.
{
    "edges": [
        {"from": "variable1", "to": "variable2", "weight": 0.85}
    ],
    "variables": ["var1", "var2", ...]
}
```

```

### Optional: keep all algorithms in catalog but restrict selection
- We preserved the catalog for role mapping but filtered selections to implemented algorithms.
- When you add `CAM`, `FCI`, or `PNL` later, add their tool wrappers and `_run_*` handlers and include them in `self.IMPLEMENTED_ALGOS`.

- I reviewed the agent and tools, identified misalignments (unsupported algorithms, ANM score interpretation, EqVar references, and README schema), and provided concise edits to fix them.
- Next, apply the above edits. This will ensure selection only includes runnable methods, ANM scoring matches its statistical interpretation, bootstrap evaluation doesn’t reference missing tools, and the README accurately reflects behavior.
```

## HITL (Human-in-the-Loop) 지원

다음 단계에서 HITL이 활성화됩니다:

- `assumption_method_matrix`: 가정 검증 결과 검토
- `algorithm_scoring`: 알고리즘 선택 검토
- `final_graph_selection`: 최종 그래프 선택 검토

## 오류 처리

각 단계에서 오류가 발생하면:

1. 오류가 `state["error"]`에 기록됩니다
2. 해당 단계의 상태가 "failed"로 설정됩니다
3. 로그에 상세한 오류 정보가 기록됩니다

## 성능 고려사항

- **병렬 처리**: 알고리즘 실행은 ThreadPoolExecutor로 병렬화됩니다
- **메모리**: 대용량 데이터의 경우 배치 처리 고려
- **시간**: 부트스트랩 반복 횟수와 교차검증 폴드 수 조정

## 예제

```python
# 간단한 예제
import pandas as pd
import numpy as np
from agents.causal_discovery.agent import CausalDiscoveryAgent

# 샘플 데이터 생성
data = pd.DataFrame({
    'X': np.random.normal(0, 1, 100),
    'Y': 2 * np.random.normal(0, 1, 100) + 0.5 * np.random.normal(0, 1, 100),
    'Z': 3 * np.random.normal(0, 1, 100) + 0.3 * np.random.normal(0, 1, 100)
})

# Agent 생성 및 실행
agent = CausalDiscoveryAgent()
state = {"df_preprocessed": data, "current_substep": "assumption_method_matrix"}

# 전체 파이프라인 실행
for substep in ["assumption_method_matrix", "algorithm_scoring",
                "run_algorithms", "intermediate_scoring", "final_graph_selection"]:
    state["current_substep"] = substep
    state = agent.step(state)
    print(f"Completed {substep}: {state.get('causal_discovery_status', 'unknown')}")

# 결과 확인
print("Selected graph:", state.get("selected_graph"))
print("Selection reasoning:", state.get("graph_selection_reasoning"))
```
