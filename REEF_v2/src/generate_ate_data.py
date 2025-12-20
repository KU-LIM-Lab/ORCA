"""
REEF 데이터를 이용한 ATE 측정 데이터 생성 스크립트
"""

import json
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
import yaml

try:
    from .ate_calculator import calculate_ate
    from .reef_data_loader import REEFDataLoader
except ImportError:
    # 직접 실행 시 또는 모듈로 import 시
    import sys
    from pathlib import Path
    current_dir = Path(__file__).parent
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
    try:
        from ate_calculator import calculate_ate
        from reef_data_loader import REEFDataLoader
    except ImportError:
        # 프로젝트 루트에서 실행하는 경우
        project_root = current_dir.parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        from REEF_v2.src.ate_calculator import calculate_ate
        from REEF_v2.src.reef_data_loader import REEFDataLoader


def load_queries_from_yaml(yaml_path: str) -> List[Dict[str, Any]]:
    """
    YAML 파일에서 쿼리 목록을 로드합니다.
    
    Args:
        yaml_path: YAML 파일 경로
    
    Returns:
        쿼리 딕셔너리 리스트
    """
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    return data.get('queries', [])


def generate_question(treatment: str, outcome: str) -> str:
    """
    treatment와 outcome으로부터 질문을 생성합니다.
    
    Args:
        treatment: treatment 변수명
        outcome: outcome 변수명
    
    Returns:
        질문 문자열
    """
    # 변수명에서 테이블 prefix 제거
    treatment_clean = treatment.split('.')[-1] if '.' in treatment else treatment
    outcome_clean = outcome.split('.')[-1] if '.' in outcome else outcome
    
    return f"What is the causal effect of {treatment_clean} on {outcome_clean}?"


def resolve_variable_name(
    var_name: str,
    df: pd.DataFrame,
    table_prefix: Optional[str] = None
) -> str:
    """
    변수명을 데이터프레임의 실제 컬럼명으로 해석합니다.
    
    Args:
        var_name: 변수명 (예: "unit_price" 또는 "order_items.unit_price")
        df: 데이터프레임
        table_prefix: 테이블 prefix (예: "order_items")
    
    Returns:
        실제 컬럼명
    """
    # 이미 테이블 prefix가 있으면 그대로 사용
    if '.' in var_name:
        # 테이블 prefix와 컬럼명 분리
        parts = var_name.split('.')
        if len(parts) == 2:
            table, col = parts
            # 데이터프레임에서 찾기
            if f"{table}.{col}" in df.columns:
                return f"{table}.{col}"
            elif col in df.columns:
                return col
    
    # 테이블 prefix가 주어진 경우
    if table_prefix:
        full_name = f"{table_prefix}.{var_name}"
        if full_name in df.columns:
            return full_name
    
    # 직접 매칭
    if var_name in df.columns:
        return var_name
    
    # 부분 매칭 시도
    for col in df.columns:
        if col.endswith(f".{var_name}") or col == var_name:
            return col
    
    raise ValueError(f"Variable '{var_name}' not found in dataframe columns: {list(df.columns)}")


def process_single_query(
    query_config: Dict[str, Any],
    loader: REEFDataLoader,
    output_dir: Optional[Path] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    단일 쿼리를 처리하여 ATE를 계산합니다.
    
    Args:
        query_config: 쿼리 설정 딕셔너리
        loader: REEF 데이터 로더
        output_dir: 출력 디렉토리 (None이면 출력 안 함)
        verbose: 상세 출력 여부
    
    Returns:
        결과 딕셔너리
    """
    treatment = query_config.get('treatment')
    outcome = query_config.get('outcome')
    confounders = query_config.get('confounders', [])
    mediators = query_config.get('mediators', [])
    instrumental_variables = query_config.get('instrumental_variables', [])
    sql_query = query_config.get('sql_query')
    table_name = query_config.get('table_name')
    limit = query_config.get('limit')
    question = query_config.get('question')
    estimator = query_config.get('estimator')  # None이면 자동 선택
    
    if verbose:
        print(f"\nProcessing: {treatment} -> {outcome}")
    
    # 데이터 로드
    try:
        if sql_query:
            df = loader.load_custom_query(sql_query)
        elif table_name:
            df = loader.load_table(table_name, limit=limit)
        else:
            raise ValueError("Either 'sql_query' or 'table_name' must be provided")
        
        if len(df) == 0:
            raise ValueError("Loaded dataframe is empty")
        
        if verbose:
            print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
        
    except Exception as e:
        error_msg = f"Failed to load data: {e}"
        if verbose:
            print(f"  ERROR: {error_msg}")
        return {
            "question": question or generate_question(treatment, outcome),
            "treatment": treatment,
            "outcome": outcome,
            "confounders": confounders,
            "mediators": mediators,
            "instrumental_variables": instrumental_variables,
            "error": error_msg
        }
    
    # 변수명 해석
    try:
        treatment_resolved = resolve_variable_name(treatment, df)
        outcome_resolved = resolve_variable_name(outcome, df)
        confounders_resolved = [resolve_variable_name(c, df) for c in confounders]
        mediators_resolved = [resolve_variable_name(m, df) for m in mediators]
        ivs_resolved = [resolve_variable_name(iv, df) for iv in instrumental_variables]
        
        if verbose:
            print(f"  Treatment: {treatment} -> {treatment_resolved}")
            print(f"  Outcome: {outcome} -> {outcome_resolved}")
            if confounders_resolved:
                print(f"  Confounders: {confounders_resolved}")
        
    except Exception as e:
        error_msg = f"Failed to resolve variable names: {e}"
        if verbose:
            print(f"  ERROR: {error_msg}")
        return {
            "question": question or generate_question(treatment, outcome),
            "treatment": treatment,
            "outcome": outcome,
            "confounders": confounders,
            "mediators": mediators,
            "instrumental_variables": instrumental_variables,
            "error": error_msg
        }
    
    # ATE 계산
    try:
        result = calculate_ate(
            df=df,
            treatment=treatment_resolved,
            outcome=outcome_resolved,
            confounders=confounders_resolved,
            mediators=mediators_resolved,
            instrumental_variables=ivs_resolved,
            estimator=estimator
        )
        
        # 결과 정리
        output = {
            "question": question or generate_question(treatment, outcome),
            "treatment": treatment,
            "outcome": outcome,
            "confounders": confounders,
            "mediators": mediators,
            "instrumental_variables": instrumental_variables,
            "ground_truth_ate": result["ate"],
            "confidence_interval": result["confidence_interval"],
            "p_value": result["p_value"],
            "estimation_method": result["estimator"],
            "treatment_type": result["treatment_type"],
            "outcome_type": result["outcome_type"],
            "n_samples": result["n_samples"]
        }
        
        if verbose:
            print(f"  ATE: {result['ate']}")
            if result['confidence_interval']:
                print(f"  CI: {result['confidence_interval']}")
            print(f"  Estimator: {result['estimator']}")
        
        return output
        
    except Exception as e:
        error_msg = f"Failed to calculate ATE: {e}"
        if verbose:
            print(f"  ERROR: {error_msg}")
        return {
            "question": question or generate_question(treatment, outcome),
            "treatment": treatment,
            "outcome": outcome,
            "confounders": confounders,
            "mediators": mediators,
            "instrumental_variables": instrumental_variables,
            "error": error_msg
        }


def generate_ate_data(
    queries: List[Dict[str, Any]],
    db_name: str = "reef_db",
    output_path: Optional[str] = None,
    verbose: bool = True
) -> List[Dict[str, Any]]:
    """
    여러 쿼리에 대해 ATE 데이터를 생성합니다.
    
    Args:
        queries: 쿼리 설정 리스트
        db_name: 데이터베이스 이름
        output_path: 출력 파일 경로 (None이면 출력 안 함)
        verbose: 상세 출력 여부
    
    Returns:
        결과 딕셔너리 리스트
    """
    loader = REEFDataLoader(db_name=db_name)
    results = []
    
    if verbose:
        print(f"Processing {len(queries)} queries...")
    
    for i, query_config in enumerate(queries, 1):
        if verbose:
            print(f"\n[{i}/{len(queries)}]")
        
        result = process_single_query(query_config, loader, verbose=verbose)
        results.append(result)
    
    # 결과 저장
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        if verbose:
            print(f"\nResults saved to {output_path}")
    
    # 통계 출력
    if verbose:
        successful = sum(1 for r in results if "error" not in r)
        failed = len(results) - successful
        print(f"\nSummary: {successful} successful, {failed} failed")
    
    return results


def main():
    """CLI 진입점"""
    parser = argparse.ArgumentParser(description="Generate ATE data from REEF database")
    # parser.add_argument(
    #     "--quiet",
    #     action="store_true",
    #     help="Suppress verbose output"
    # )
    
    args = parser.parse_args()
    
    # # 출력 경로 설정
    # if args.output is None:
    #     queries_path = Path(args.queries)
    #     args.output = str(queries_path.with_suffix('.json'))
    
    # 쿼리 로드
    queries = load_queries_from_yaml("REEF_v2/configs/ate_queries.yaml")
    
    # 데이터 생성
    results = generate_ate_data(
        queries=queries,
        db_name="reef_db",
        output_path="REEF_v2/outputs/ate_results.json",
        verbose= True
    )
    
    return 0 if all("error" not in r for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())

