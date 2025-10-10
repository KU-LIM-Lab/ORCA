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
            "run_all_tier_algorithms": False
        }
    )
    logger.info("CausalDiscoveryAgent created successfully")
    
    # 3. 초기 상태 생성 (기본 실행)
    state = {
        "df_preprocessed": df,
        "current_substep": "data_profiling",
        "initial_query": "Test causal discovery",
        "db_id": "test_db"
    #   "run_all_tier_algorithms": True  # 계열 전체 실행
    }
    
    # 4. 새로운 파이프라인 단계별 실행
    substeps = [
        "data_profiling",
        "algorithm_tiering", 
        "run_algorithms_portfolio",
        "candidate_pruning",
        "scorecard_evaluation",
        "ensemble_synthesis"
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
                if substep == "data_profiling":
                    profile = state.get("data_profile", {})
                    scores = state.get("assumption_method_scores", {})
                    logger.info(f"  - Data profile: {profile.get('summary', 'N/A')}")
                    logger.info(f"  - Generated {len(scores)} assumption types")
                    for score_type, score_dict in scores.items():
                        logger.info(f"    {score_type}: {len(score_dict)} variable pairs")
                
                elif substep == "algorithm_tiering":
                    tiers = state.get("algorithm_tiers", {})
                    reasoning = state.get("tiering_reasoning", "")
                    logger.info(f"  - Tier 1 algorithms: {tiers.get('tier1', [])}")
                    logger.info(f"  - Tier 2 algorithms: {tiers.get('tier2', [])}")
                    logger.info(f"  - Tier 3 algorithms: {tiers.get('tier3', [])}")
                    logger.info(f"  - Tiering reasoning: {reasoning[:100]}...")
                
                elif substep == "run_algorithms_portfolio":
                    results = state.get("algorithm_results", {})
                    logger.info(f"  - Executed {len(results)} algorithms")
                    for alg, result in results.items():
                        if "error" in result:
                            logger.warning(f"    {alg}: Error - {result['error']}")
                        else:
                            n_edges = len(result.get("graph", {}).get("edges", []))
                            logger.info(f"    {alg}: {n_edges} edges found")
                
                elif substep == "candidate_pruning":
                    pruned = state.get("pruned_candidates", [])
                    pruning_log = state.get("pruning_log", [])
                    logger.info(f"  - Pruned candidates: {len(pruned)}")
                    logger.info(f"  - Rejected candidates: {len(pruning_log)}")
                    for candidate in pruned:
                        alg = candidate.get("algorithm", "Unknown")
                        violation = candidate.get("violation_ratio", 0)
                        instability = candidate.get("instability_score", 0)
                        logger.info(f"    {alg}: violation={violation:.3f}, instability={instability:.3f}")
                
                elif substep == "scorecard_evaluation":
                    scorecard = state.get("scorecard", [])
                    top_candidates = state.get("top_candidates", [])
                    logger.info(f"  - Scorecard entries: {len(scorecard)}")
                    logger.info(f"  - Top candidates: {len(top_candidates)}")
                    for candidate in top_candidates:
                        alg = candidate.get("algorithm", "Unknown")
                        composite = candidate.get("composite_score", 0)
                        logger.info(f"    {alg}: composite_score={composite:.3f}")
                
                elif substep == "ensemble_synthesis":
                    consensus_pag = state.get("consensus_pag", {})
                    selected_graph = state.get("selected_graph", {})
                    reasoning = state.get("synthesis_reasoning", "")
                    status = state.get("causal_discovery_status", "unknown")
                    logger.info(f"  - Final status: {status}")
                    logger.info(f"  - Consensus PAG edges: {len(consensus_pag.get('edges', []))}")
                    logger.info(f"  - Selected DAG edges: {len(selected_graph.get('edges', []))}")
                    logger.info(f"  - Synthesis reasoning: {reasoning[:100]}...")
                    
        except Exception as e:
            logger.error(f"Exception in {substep}: {str(e)}")
            break
    
    # 5. 최종 결과 요약
    logger.info("\n=== Final Results ===")
    logger.info(f"Causal discovery status: {state.get('causal_discovery_status', 'unknown')}")
    
    # 데이터 프로파일 출력
    if "data_profile" in state:
        profile = state["data_profile"]
        logger.info(f"Data profile: {profile.get('summary', 'N/A')}")
    
    # 알고리즘 계층 출력
    if "algorithm_tiers" in state:
        tiers = state["algorithm_tiers"]
        logger.info(f"Algorithm tiers: Tier1={tiers.get('tier1', [])}, Tier2={tiers.get('tier2', [])}, Tier3={tiers.get('tier3', [])}")
    
    # 최종 그래프 출력
    if "selected_graph" in state:
        graph = state["selected_graph"]
        edges = graph.get("edges", [])
        logger.info(f"Selected DAG has {len(edges)} edges:")
        for edge in edges:
            logger.info(f"  {edge.get('from', '?')} -> {edge.get('to', '?')} (weight: {edge.get('weight', 0):.3f})")
    
    # 합의 PAG 출력
    if "consensus_pag" in state:
        pag = state["consensus_pag"]
        edges = pag.get("edges", [])
        logger.info(f"Consensus PAG has {len(edges)} edges:")
        for edge in edges:
            direction = edge.get("direction", "unknown")
            marker = edge.get("marker", "?")
            confidence = edge.get("confidence", 0)
            logger.info(f"  {edge.get('from', '?')} {marker} {edge.get('to', '?')} (direction: {direction}, confidence: {confidence:.3f})")
    
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
