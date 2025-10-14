# examples/orchestration_flow.py
"""
Complete orchestration flow example showing how the system works
"""
from orchestration.graph import create_orchestration_graph
from monitoring.metrics.collector import MetricsCollector, set_metrics_collector
from core.state import create_initial_state
import os

def main():
    """Demonstrate the complete orchestration flow"""
    
    # 1. Initialize metrics collector
    collector = MetricsCollector("orchestration_demo")
    set_metrics_collector(collector)
    collector.start_monitoring()
    
    # 2. Create orchestration graph
    graph = create_orchestration_graph(
        metrics_collector=collector,
        orchestration_config={"interactive": False}
    )
    
    # 3. Example 1: Full pipeline (default behavior)
    print("=== Example: Full Pipeline ===")
    # execute할 때 config에서 db_id 받아와서 쓸 수 있도록 하기.. or main에서 미리 지정해주기 ..
    result1 = graph.execute("What is the causal effect of gender on used_coupon?")
    print(f"Status: {result1.get('execution_status')}")
    print(f"Steps completed: {len(result1.get('execution_log', []))}")
    
    fr = result1.get("final_report", {})
    print("\n=== Final Report Summary ===")
    print(f"query: {fr.get('query')}")
    print(f"status: {fr.get('status')}")
    print(f"total_steps: {fr.get('total_steps')}")

    print("\n=== Final Report (Markdown) ===")
    print(fr.get("markdown", ""))
    
    
    # 6. Show execution flow
    print("\n=== Execution Flow Analysis ===")
    show_execution_flow(result1)
    
    # 7. Stop monitoring and show metrics
    collector.stop_monitoring()
    show_metrics_summary(collector)

    # 8. Optional: Interactive mode demo (set ORCA_DEMO_INTERACTIVE=1 to enable)
    if os.environ.get("ORCA_DEMO_INTERACTIVE") == "1":
        print("\n=== Example: Interactive Mode (will prompt) ===")
        collector2 = MetricsCollector("orchestration_demo_interactive")
        set_metrics_collector(collector2)
        collector2.start_monitoring()
        graph_interactive = create_orchestration_graph(
            metrics_collector=collector2,
            orchestration_config={"interactive": True}
        )
        try:
            result_int = graph_interactive.execute("Run interactive causal pipeline demo")
            print(f"Interactive Status: {result_int.get('execution_status')}")
            print(f"Interactive Final report generated: {'final_report' in result_int}")
        finally:
            collector2.stop_monitoring()
            show_metrics_summary(collector2)

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
    checks = [
        ("data_exploration", result.get("data_exploration_status") == "completed" \
            or bool(result.get("df_preprocessed")) or bool(result.get("selected_tables"))),
        ("selected_tables", bool(result.get("selected_tables"))),
        ("causal_graph", bool(result.get("selected_graph"))),
        ("algorithm_scores", bool(result.get("algorithm_scores"))),
        ("causal_estimates", bool(result.get("causal_estimates"))),
        ("confidence_intervals", bool(result.get("confidence_intervals"))),
        ("final_report", bool(result.get("final_report"))),
    ]
    for label, ok in checks:
        print(f"  {'✓' if ok else '✗'} {label}: {'Available' if ok else 'Not available'}")

    # Planned vs completed steps
    print("\nPlan Status:")
    total_steps = len(result.get("execution_plan", [])) if isinstance(result.get("execution_plan"), list) else result.get("total_steps", 0)
    completed_substeps = result.get("completed_substeps", []) or []
    current_idx = result.get("current_execute_step", 0)
    print(f"  planned steps: {total_steps}")
    print(f"  completed substeps: {len(completed_substeps)}")
    print(f"  current pointer: {current_idx}")
    if result.get("executor_completed"):
        print("  ✅ executor: completed")
    else:
        print("  ⏳ executor: in progress")

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
