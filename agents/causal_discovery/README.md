# CausalDiscoveryAgent 사용법

## 개요

CausalDiscoveryAgent는 인과관계 발견을 위한 전문 에이전트입니다. 새로운 5단계 파이프라인을 통해 데이터 프로파일링, 알고리즘 계층화, 포트폴리오 실행, 후보 정제, 앙상블 합성을 수행합니다.

## 새로운 파이프라인 구조 (5단계)

### Stage 1: Data Profiling & Algorithm Tiering

#### 1.1 data_profiling

- **목적**: 데이터 특성 프로파일링 및 정성적 요약 생성
- **출력**: `data_profile` (선형성, 비가우시안성, ANM 호환성 등), `assumption_method_scores`

#### 1.2 algorithm_tiering

- **목적**: 데이터 프로파일에 기반한 알고리즘 계층화 (계열당 대표 알고리즘 선택)
- **전략**:
  - 각 계열(family)에서 1개의 대표 알고리즘만 기본 선택
  - Nonlinear-FCM 계열: ANM을 기본으로, PNL/CAM은 명시적 요청시만 포함
  - FCI는 항상 포함 (잠재 변수 강건성)
- **입력**: `data_profile` (+ 선택적으로 `run_all_tier_algorithms` via config/state)
- **출력**: `algorithm_tiers` (Tier1: 최적 매치, Tier2: 부분 매치, Tier3: 탐색용), `tiering_reasoning`

### Stage 2: Parallel Algorithm Execution

#### 2.1 run_algorithms_portfolio

- **목적**: 모든 계층의 알고리즘을 병렬로 실행
- **출력**: `algorithm_results` (모든 알고리즘의 그래프와 메타데이터)

### Stage 3: Candidate Pruning

#### 3.1 candidate_pruning

- **목적**: CI 테스트와 구조적 일관성을 통한 후보 정제
- **출력**: `pruned_candidates`, `pruning_log`

### Stage 4: Scorecard Evaluation

#### 4.1 scorecard_evaluation

- **목적**: 복합 점수카드를 통한 후보 평가
- **출력**: `scorecard`, `top_candidates` (상위 3개 그래프)

### Stage 5: Ensemble & Synthesis

#### 5.1 ensemble_synthesis

- **목적**: PAG-like 및 DAG 출력을 통한 앙상블 합성
- **출력**: `consensus_pag` (보고용), `selected_graph` (추론용), `synthesis_reasoning`

## 알고리즘 패밀리 분류

```python
ALGORITHM_FAMILIES = {
    "Linear": ["LiNGAM"],
    "Nonlinear-FCM": ["ANM", "PNL", "CAM"],
    "Constraint-based": ["PC"],
    "Score-based": ["GES"],
    "Latent-robust": ["FCI"]
}
```

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
        "ci_alpha": 0.05,
        "violation_threshold": 0.1,
        "n_subsets": 3,
        "composite_weights": {
            "markov_consistency": 0.4,
            "sampling_stability": 0.3,
            "structural_stability": 0.3
        },
        # 대표 vs 전체 실행 스위치
        "run_all_tier_algorithms": False
    }
)

# 상태 생성 (전처리된 데이터 필요)
state = create_initial_state("분석 쿼리", "db_id")
state["df_preprocessed"] = your_dataframe  # pandas DataFrame
state["current_substep"] = "data_profiling"

# 단계별 실행
state = agent.step(state)
```

### 전체 파이프라인 실행

```python
# 새로운 5단계 파이프라인
substeps = [
    "data_profiling",
    "algorithm_tiering",
    "run_algorithms_portfolio",
    "candidate_pruning",
    "scorecard_evaluation",
    "ensemble_synthesis"
]

for substep in substeps:
    state["current_substep"] = substep
    state = agent.step(state)
    print(f"Completed {substep}: {state.get('causal_discovery_status', 'unknown')}")
```

## 새로운 설정 옵션

```python
config = {
    # 기존 설정
    "bootstrap_iterations": 100,
    "cv_folds": 5,
    # legacy removed: lambda_soft_and, beta_conservative

    # 새로운 설정
    "ci_alpha": 0.05,                    # CI 테스트 유의수준
    "violation_threshold": 0.1,          # CI 위반 임계값
    "n_subsets": 5,                      # 구조적 일관성 테스트용 서브셋 수
    "composite_weights": {               # 복합 점수 가중치
        "global_consistency": 0.4,
        "sampling_stability": 0.3,
        "structural_stability": 0.3
    }
}
```

## 새로운 도구 (Tools)

### PruningTool

- `global_markov_test()`: 전역 마르코프 성질 CI 테스트
- `structural_consistency_test()`: 서브샘플링을 통한 구조적 일관성 테스트

### EnsembleTool

- `build_consensus_skeleton()`: 신뢰도 점수가 있는 합의 스켈레톤 구축
- `resolve_directions()`: 불확실성 마커를 통한 방향 해결
- `construct_pag()`: PAG-like 그래프 구축
- `construct_dag()`: 가정 기반 타이 브레이킹을 통한 단일 DAG 구축

### 기존 도구

- `StatsTool`: 통계 테스트 도구
- `IndependenceTool`: 독립성 테스트 도구
- `LiNGAMTool`, `ANMTool`, `PCTool`, `GESTool`, `CAMTool`, `FCITool`: 알고리즘 도구
- `Bootstrapper`, `GraphEvaluator`, `GraphOps`: 평가 도구

## 출력 데이터

### data_profile

```python
{
    "linearity": "strong|moderate|weak",
    "non_gaussian": "strong|moderate|weak",
    "anm_compatible": bool,
    "gaussian": "strong|moderate|weak",
    "equal_variance": "strong|moderate|weak",
    "n_variables": int,
    "n_pairs": int,
    "summary": "Data shows ... patterns"
}
```

### algorithm_tiers

```python
{
    "tier1": ["LiNGAM", "PC"],           # 최적 매치 알고리즘
    "tier2": ["ANM", "CAM"],             # 부분 매치 알고리즘
    "tier3": ["GES"]                     # 탐색용 알고리즘
}
```

### scorecard

```python
[
    {
        "algorithm": "LiNGAM",
        "graph_id": "LiNGAM_12345",
        "markov_consistency": 0.90,
        "sampling_stability": 0.80,
        "structural_stability": 0.75,
        "composite_score": 0.82
    },
    ...
]
```

### consensus_pag (보고용)

```python
{
    "graph_type": "PAG",
    "edges": [
        {
            "from": "X", "to": "Y",
            "direction": "forward|backward|uncertain|conflict",
            "marker": "->|o-o|o->",
            "confidence": 0.85,
            "forward_votes": 3,
            "backward_votes": 1
        }
    ],
    "metadata": {
        "construction_method": "consensus",
        "uncertainty_markers": True
    }
}
```

### selected_graph (추론용)

```python
{
    "graph_type": "DAG",
    "edges": [
        {
            "from": "X", "to": "Y",
            "direction": "forward",
            "marker": "->",
            "tie_breaking": "Linear algorithm preference (LiNGAM)"
        }
    ],
    "metadata": {
        "construction_method": "tie_breaking",
        "top_algorithm": "LiNGAM"
    }
}
```

## HITL (Human-in-the-Loop) 지원

다음 단계에서 HITL이 활성화됩니다:

- `data_profiling`: 데이터 프로파일 검토
- `algorithm_tiering`: 알고리즘 계층화 검토
- `run_algorithms_portfolio`: 알고리즘 실행 검토
- `candidate_pruning`: 후보 정제 검토
- `scorecard_evaluation`: 점수카드 평가 검토
- `ensemble_synthesis`: 앙상블 합성 검토

## Redis 저장소 패턴

대용량 객체는 Redis에 저장되고 상태에는 키만 전달됩니다:

```python
# 알고리즘 결과가 많을 때
if len(algorithm_results) > 5:
    redis_key = f"{db_id}:algorithm_results:{session_id}"
    redis_client.set(redis_key, json.dumps(algorithm_results))
    state["algorithm_results_key"] = redis_key
```

## 예제

```python
# 새로운 파이프라인 예제
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

# 기본 실행 (계열당 대표 알고리즘만)
state = {"df_preprocessed": data, "current_substep": "data_profiling"}

# 확장 알고리즘을 명시적으로 포함하려면:
# 대표 vs 전체 실행 스위치 (config or state)
# config: {"run_all_tier_algorithms": true}
# state["run_all_tier_algorithms"] = True

# 새로운 5단계 파이프라인 실행
substeps = [
    "data_profiling",
    "algorithm_tiering",
    "run_algorithms_portfolio",
    "candidate_pruning",
    "scorecard_evaluation",
    "ensemble_synthesis"
]

for substep in substeps:
    state["current_substep"] = substep
    state = agent.step(state)
    print(f"Completed {substep}: {state.get('causal_discovery_status', 'unknown')}")

# 결과 확인
print("Data profile:", state.get("data_profile"))
print("Algorithm tiers:", state.get("algorithm_tiers"))
print("Consensus PAG:", state.get("consensus_pag"))
print("Selected DAG:", state.get("selected_graph"))
print("Synthesis reasoning:", state.get("synthesis_reasoning"))
```

## 성능 고려사항

- **병렬 처리**: 알고리즘 실행은 ThreadPoolExecutor로 병렬화됩니다
- **메모리**: 대용량 객체는 Redis에 저장하여 메모리 사용량 최적화
- **시간**: CI 테스트와 구조적 일관성 테스트로 인한 추가 시간 고려
- **확장성**: 새로운 알고리즘 패밀리와 도구 추가 가능
