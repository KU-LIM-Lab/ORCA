# examples/orchestration_flow.py
"""
Complete orchestration flow example showing how the system works
"""
from orchestration.graph import create_orchestration_graph
from monitoring.metrics.collector import MetricsCollector, set_metrics_collector
from core.state import create_initial_state

def main():
    """Demonstrate the complete orchestration flow"""
    
    # 1. Initialize metrics collector
    collector = MetricsCollector("orchestration_demo")
    set_metrics_collector(collector)
    collector.start_monitoring()
    
    # 2. Create orchestration graph
    graph = create_orchestration_graph(metrics_collector=collector)
    
    # 3. Example 1: Full pipeline (default behavior)
    print("=== Example 1: Full Pipeline ===")
    # execute할 때 config에서 db_id 받아와서 쓸 수 있도록 하기.. or main에서 미리 지정해주기 ..
    result1 = graph.execute("고객 이탈에 영향을 미치는 요인을 분석해줘")
    print(f"Status: {result1.get('execution_status')}")
    print(f"Steps completed: {len(result1.get('execution_log', []))}")
    print(f"Final report generated: {'final_report' in result1}")
    
    # 4. Example 2: Resume from middle (data already explored)
    print("\n=== Example 2: Resume from Causal Discovery ===")
    state_with_data = {
        "initial_query": "인과 그래프를 발견해줘",
        "current_state": {
            "data_explored": True,
            "schema_metadata": {"tables": ["customers", "orders"]},
            "candidate_tables": ["customers", "orders", "products"]
        }
    }
    result2 = graph.execute("인과 그래프를 발견해줘", state_with_data)
    print(f"Status: {result2.get('execution_status')}")
    print(f"Steps completed: {len(result2.get('execution_log', []))}")
    
    # 5. Example 3: Only causal inference (previous steps completed)
    print("\n=== Example 3: Only Causal Inference ===")
    state_with_graph = {
        "initial_query": "인과 효과를 추정해줘",
        "current_state": {
            "data_explored": True,
            "schema_metadata": {"tables": ["customers", "orders"]},
            "candidate_tables": ["customers", "orders", "products"],
            "causal_graph": {"nodes": ["A", "B", "C"], "edges": [("A", "B"), ("B", "C")]},
            "algorithm_scores": {"PC": 0.8, "GES": 0.7}
        }
    }
    result3 = graph.execute("인과 효과를 추정해줘", state_with_graph)
    print(f"Status: {result3.get('execution_status')}")
    print(f"Steps completed: {len(result3.get('execution_log', []))}")
    
    # 6. Show execution flow
    print("\n=== Execution Flow Analysis ===")
    show_execution_flow(result1)
    
    # 7. Stop monitoring and show metrics
    collector.stop_monitoring()
    show_metrics_summary(collector)

def show_execution_flow(result):
    """Show the execution flow from the result"""
    execution_log = result.get("execution_log", [])
    
    print("Execution Flow:")
    for i, log in enumerate(execution_log, 1):
        status = "✅" if log.get("success", False) else "❌"
        duration = log.get("duration", 0)
        print(f"  {i}. {log.get('step_id', 'unknown')} - {status} ({duration:.2f}s)")
    
    # Show state progression
    print("\nState Progression:")
    state_keys = [
        "data_explored", "schema_metadata", "candidate_tables",
        "causal_graph", "algorithm_scores", 
        "causal_estimates", "confidence_intervals",
        "final_report"
    ]
    
    for key in state_keys:
        if key in result:
            print(f"  ✓ {key}: Available")
        else:
            print(f"  ✗ {key}: Not available")

def show_metrics_summary(collector):
    """Show metrics summary"""
    summary = collector.get_metrics_summary()
    
    print("\n=== Metrics Summary ===")
    print(f"Total metrics: {summary['total_metrics']}")
    print(f"Session duration: {summary['session_duration']:.2f}s")
    
    if 'execution_time' in summary['by_type']:
        exec_time = summary['by_type']['execution_time']
        print(f"Average execution time: {exec_time['average']:.2f}s")
        print(f"Total execution time: {exec_time['total']:.2f}s")
    
    if 'token_count' in summary['by_type']:
        tokens = summary['by_type']['token_count']
        print(f"Total tokens used: {tokens['total']:.0f}")

if __name__ == "__main__":
    main()
