import sys
import os
from pprint import pprint

# Ensure project root on sys.path when running directly
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from orchestration.graph import OrchestrationGraph
from core.state import create_initial_state


def main():
    graph = OrchestrationGraph()

    # ===== Test A: Data exploration (system steps execution) =====
    init = create_initial_state(query="tell me about the user table")
    init.update({
        "analysis_mode": "data_exploration"
    })

    state_after_planner = graph._planner_node(init)
    plan = state_after_planner.get("execution_plan", [])
    print("[A][Planner] Plan length:", len(plan))
    print("[A][Planner] Plan steps (phase/substep, agent):")
    for i, s in enumerate(plan, 1):
        print(f"  {i:02d}. {s['phase'].value} / {s['substep']}  agent={s.get('agent')}")
    if not plan:
        print("[A][Planner] Empty plan; falling back to system steps.")
        plan = [s for s in graph.planner.get_full_pipeline_plan() if s.get("is_system_component", False)]

    system_plan = [s for s in plan if s.get("is_system_component", False)]
    state_after_planner["execution_plan"] = system_plan
    result = graph.executor.execute_plan(state_after_planner)
    print("[A][Executor] success:", result.success)
    if not result.success:
        print("[A][Executor] error:", result.error)
        sys.exit(1)

    state_after_planner.update(result.data or {})
    print("[A][State] Keys after execution:")
    for k in ["database_connection", "schema_info", "table_metadata", "table_relations"]:
        print(f"  {k}:", "present" if state_after_planner.get(k) else "missing")

    # ===== Test B: Full pipeline (plan coverage; execute only system steps) =====
    init_full = create_initial_state(query="How does days registered relate to purchase amount?")
    init_full.update({
        "analysis_mode": "full_pipeline"
    })
    state_full = graph._planner_node(init_full)
    full_plan = state_full.get("execution_plan", [])
    print("[B][Planner] Full plan length:", len(full_plan))
    print("[B][Planner] Full plan steps (phase/substep, agent):")
    for i, s in enumerate(full_plan, 1):
        print(f"  {i:02d}. {s['phase'].value} / {s['substep']}  agent={s.get('agent')}")

    # Execute only system steps to validate infra readiness
    system_full = [s for s in full_plan if s.get("is_system_component", False)] or [
        s for s in graph.planner.get_full_pipeline_plan() if s.get("is_system_component", False)
    ]
    state_full["execution_plan"] = system_full
    result_full = graph.executor.execute_plan(state_full)
    print("[B][Executor] system-only success:", result_full.success)
    if not result_full.success:
        print("[B][Executor] error:", result_full.error)
        sys.exit(1)

    print("[OK] Full pipeline plan generated; system steps executed.")


if __name__ == "__main__":
    main()


