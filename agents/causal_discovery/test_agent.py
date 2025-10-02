# agents/causal_discovery/test_agent.py
"""
CausalDiscoveryAgent 테스트 스크립트

이 스크립트는 CausalDiscoveryAgent의 기본 기능을 테스트합니다.
"""

import numpy as np
import pandas as pd
import logging
from agents.causal_discovery.agent import CausalDiscoveryAgent

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_sample_data(n_samples=200):
    """샘플 데이터 생성"""
    np.random.seed(42)
    
    # 인과관계가 있는 데이터 생성
    X = np.random.normal(0, 1, n_samples)
    Y = 2 * X + np.random.normal(0, 0.5, n_samples)  # X -> Y
    Z = 1.5 * Y + 0.8 * X + np.random.normal(0, 0.3, n_samples)  # X, Y -> Z
    
    df = pd.DataFrame({
        'X': X,
        'Y': Y, 
        'Z': Z
    })
    
    return df

def test_causal_discovery_agent():
    """CausalDiscoveryAgent 테스트"""
    logger.info("Starting CausalDiscoveryAgent test...")
    
    # 1. 샘플 데이터 생성
    df = create_sample_data()
    logger.info(f"Created sample data with shape: {df.shape}")
    logger.info(f"Data columns: {list(df.columns)}")
    
    # 2. Agent 생성
    agent = CausalDiscoveryAgent(
        name="test_causal_discovery",
        config={
            "bootstrap_iterations": 10,  # 테스트용으로 줄임
            "cv_folds": 3,
            "top_k_algorithms": 2,
            "lambda_soft_and": 0.7,
            "beta_conservative": 2.0
        }
    )
    logger.info("CausalDiscoveryAgent created successfully")
    
    # 3. 초기 상태 생성
    state = {
        "df_preprocessed": df,
        "current_substep": "assumption_method_matrix",
        "initial_query": "Test causal discovery",
        "db_id": "test_db"
    }
    
    # 4. 파이프라인 단계별 실행
    substeps = [
        "assumption_method_matrix",
        "algorithm_scoring", 
        "run_algorithms",
        "intermediate_scoring",
        "final_graph_selection"
    ]
    
    for substep in substeps:
        logger.info(f"\n=== Executing {substep} ===")
        state["current_substep"] = substep
        
        try:
            state = agent.step(state)
            
            if "error" in state:
                logger.error(f"Error in {substep}: {state['error']}")
                break
            else:
                logger.info(f"✓ {substep} completed successfully")
                
                # 각 단계별 결과 출력
                if substep == "assumption_method_matrix":
                    scores = state.get("assumption_method_scores", {})
                    logger.info(f"  - Generated {len(scores)} assumption types")
                    for score_type, score_dict in scores.items():
                        logger.info(f"    {score_type}: {len(score_dict)} variable pairs")
                
                elif substep == "algorithm_scoring":
                    selected = state.get("selected_algorithms", [])
                    scores = state.get("algorithm_scores", {})
                    logger.info(f"  - Selected algorithms: {selected}")
                    for alg, score_info in scores.items():
                        logger.info(f"    {alg}: {score_info.get('final_score', 0):.3f}")
                
                elif substep == "run_algorithms":
                    results = state.get("algorithm_results", {})
                    logger.info(f"  - Executed {len(results)} algorithms")
                    for alg, result in results.items():
                        if "error" in result:
                            logger.warning(f"    {alg}: Error - {result['error']}")
                        else:
                            n_edges = len(result.get("graph", {}).get("edges", []))
                            logger.info(f"    {alg}: {n_edges} edges found")
                
                elif substep == "intermediate_scoring":
                    scores = state.get("intermediate_scores", {})
                    candidates = state.get("candidate_graphs", [])
                    logger.info(f"  - Evaluated {len(scores)} algorithms")
                    logger.info(f"  - Generated {len(candidates)} candidate graphs")
                
                elif substep == "final_graph_selection":
                    selected_graph = state.get("selected_graph", {})
                    reasoning = state.get("graph_selection_reasoning", "")
                    status = state.get("causal_discovery_status", "unknown")
                    logger.info(f"  - Final status: {status}")
                    logger.info(f"  - Selected graph edges: {len(selected_graph.get('edges', []))}")
                    logger.info(f"  - Selection reasoning: {reasoning[:100]}...")
                    
        except Exception as e:
            logger.error(f"Exception in {substep}: {str(e)}")
            break
    
    # 5. 최종 결과 요약
    logger.info("\n=== Final Results ===")
    logger.info(f"Causal discovery status: {state.get('causal_discovery_status', 'unknown')}")
    
    if "selected_graph" in state:
        graph = state["selected_graph"]
        edges = graph.get("edges", [])
        logger.info(f"Final graph has {len(edges)} edges:")
        for edge in edges:
            logger.info(f"  {edge.get('from', '?')} -> {edge.get('to', '?')} (weight: {edge.get('weight', 0):.3f})")
    
    if "error" in state:
        logger.error(f"Pipeline failed with error: {state['error']}")
        return False
    else:
        logger.info("✓ CausalDiscoveryAgent test completed successfully!")
        return True

if __name__ == "__main__":
    success = test_causal_discovery_agent()
    if success:
        print("\n🎉 All tests passed!")
    else:
        print("\n❌ Tests failed!")
