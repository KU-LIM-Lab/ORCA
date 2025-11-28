# tests/orchestration_flow.py
"""
Complete example showing how the system works
"""
import os
# Fix OpenMP duplicate library error on macOS
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from orchestration.graph import create_orchestration_graph
from monitoring.metrics.collector import MetricsCollector, set_metrics_collector
from core.state import create_initial_state
import argparse

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Orchestration flow demo (step visibility)")
    parser.add_argument("--query", default=None, help="Query to run (if not provided, will prompt from terminal)")
    parser.add_argument("--interactive", default=True, action="store_true", help="Run with HITL prompts enabled")
    parser.add_argument("--verbose", action="store_true", help="Show detailed execution flow and report")
    return parser.parse_args()


def main():
    """Demonstrate the complete orchestration flow"""
    args = _parse_args()
    
    # Get query from argument or terminal input
    query = args.query
    if not query:
        print("\n🤖 ORCA Orchestration Flow Demo")
        print("=" * 60)
        print("Enter your causal analysis query (or 'exit' to quit)")
        print("Example: What is the causal effect of gender on used_coupon?")
        print("=" * 60)
        query = input("\n🧑 Query: ").strip()
        if not query or query.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            return
    
    collector = MetricsCollector("orchestration_demo")
    set_metrics_collector(collector)
    collector.start_monitoring()
    
    graph = create_orchestration_graph(
        metrics_collector=collector,
        orchestration_config={"interactive": args.interactive}
    )
    
    result = graph.execute(query)
    print(f"Status: {result.get('execution_status')}")
    print(f"Steps completed: {len(result.get('execution_log', []))}")
    
    fr = result.get("final_report", {}) or {}
    print("\n=== Final Report Summary ===")
    print(f"query: {fr.get('query')}")
    print(f"status: {fr.get('status')}")
    if args.verbose:
        print(f"total_steps: {fr.get('total_steps')}")
        print("\n=== Final Report (Markdown) ===")
        print(fr.get("markdown", ""))
    
    if args.verbose:
        print("\n=== Execution Flow ===")
        show_execution_flow(result)
    
    collector.stop_monitoring()
    show_metrics_summary(collector)

def show_execution_flow(result):
    """Show the execution flow from the result (concise)."""
    execution_log = result.get("execution_log", [])
    for i, log in enumerate(execution_log, 1):
        status = "ok" if log.get("success", False) else "fail"
        duration = float(log.get("duration", 0) or 0)
        step_id = log.get("step_id") or f"{log.get('phase','?')}/{log.get('substep','?')}"
        print(f"{i:02d}. {step_id}: {status} ({duration:.2f}s)")

def show_metrics_summary(collector):
    """Show metrics summary (concise)."""
    summary = collector.get_metrics_summary()
    print("\n=== Metrics Summary ===")
    print(f"metrics: {summary.get('total_metrics', 0)}")
    print(f"session_sec: {summary.get('session_duration', 0.0):.2f}")
    if 'execution_time' in summary.get('by_type', {}):
        exec_time = summary['by_type']['execution_time']
        print(f"exec_avg_sec: {exec_time.get('average', 0.0):.2f}")
        print(f"exec_total_sec: {exec_time.get('total', 0.0):.2f}")

if __name__ == "__main__":
    main()
