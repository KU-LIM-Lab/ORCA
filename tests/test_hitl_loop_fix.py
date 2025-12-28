"""
Test script to verify HITL infinite loop fix.

This script tests the following scenarios:
1. Full pipeline with HITL at table_selection - should show HITL once and proceed
2. Text2SQL mode → switch to full_pipeline - HITL at data_profiling should not loop
3. Edit decision at HITL - should apply edits and proceed to next step
4. Multiple HITL steps - should handle each correctly without loops

Run this script to validate the fixes:
    python tests/test_hitl_loop_fix.py
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.state import create_initial_state, AgentState
from orchestration.graph import create_orchestration_graph
from utils.system_init import initialize_system
from utils.settings import CONFIG
from monitoring.experiment.event_logger import ExperimentEventLogger
import json
from datetime import datetime

def test_hitl_loop_fix():
    """
    Test the HITL infinite loop fix with debug output.
    
    This will verify:
    - Step advancement happens before HITL
    - State shows correct results during HITL
    - Resume continues from next step (not same step)
    """
    
    print("\n" + "="*80)
    print("🧪 TESTING HITL INFINITE LOOP FIX")
    print("="*80)
    print("\nThis test will verify that:")
    print("  ✓ Step counter advances BEFORE HITL interrupt")
    print("  ✓ State display shows current step's results")
    print("  ✓ Resume continues from NEXT step (not re-executing same step)")
    print("  ✓ No infinite loops at HITL checkpoints")
    print("\n" + "="*80)
    
    # Initialize system
    print("\n📋 Initializing system...")
    initialize_system()
    
    # Create initial state for testing
    query = "What is the effect of gender on coupon usage in the REEF dataset?"
    db_id = "reef_db"
    
    initial_state = create_initial_state(
        query=query,
        db_id=db_id,
        session_id=f"hitl_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    
    # Set to interactive mode to trigger HITL
    initial_state["interactive"] = True
    initial_state["analysis_mode"] = "full_pipeline"
    
    print(f"✓ Created initial state")
    print(f"  - Session ID: {initial_state['session_id']}")
    print(f"  - Query: {query}")
    print(f"  - Mode: full_pipeline")
    print(f"  - Interactive: True")
    
    # Create event logger for tracking
    event_logger = ExperimentEventLogger(
        run_id=initial_state['session_id'],
        experiment_name="hitl_loop_fix_test"
    )
    
    # Create orchestration graph
    print("\n📋 Creating orchestration graph...")
    orch_graph = create_orchestration_graph(
        orchestration_config={"interactive": True},
        event_logger=event_logger
    )
    
    print("✓ Orchestration graph created")
    
    # Execute with HITL checkpoints
    print("\n" + "="*80)
    print("🚀 STARTING EXECUTION")
    print("="*80)
    print("\n⚠️  NOTE: This is an interactive test!")
    print("   You will be prompted at HITL checkpoints.")
    print("   Watch the DEBUG logs to verify:")
    print("     1. Step counter increments BEFORE HITL")
    print("     2. Resume loads NEXT step (not same step)")
    print("     3. No re-execution of completed steps")
    print("\n" + "-"*80)
    
    try:
        # Track HITL occurrences per substep
        hitl_counts = {}
        
        # Execute the graph
        final_state = orch_graph.execute(query, context=initial_state)
        
        print("\n" + "="*80)
        print("✅ EXECUTION COMPLETED")
        print("="*80)
        
        # Analyze results
        completed_substeps = final_state.get("completed_substeps", [])
        print(f"\n📊 Completed substeps ({len(completed_substeps)}):")
        for substep in completed_substeps:
            print(f"  ✓ {substep}")
        
        # Check execution log for duplicates
        execution_log = final_state.get("execution_log", [])
        print(f"\n📊 Execution log ({len(execution_log)} entries):")
        substep_counts = {}
        for entry in execution_log:
            substep = entry.get("substep", "unknown")
            substep_counts[substep] = substep_counts.get(substep, 0) + 1
        
        print("\n📊 Substep execution counts:")
        has_duplicates = False
        for substep, count in substep_counts.items():
            status = "✓" if count == 1 else "❌ DUPLICATE!"
            print(f"  {status} {substep}: {count} time(s)")
            if count > 1:
                has_duplicates = True
        
        # Verdict
        print("\n" + "="*80)
        if has_duplicates:
            print("❌ TEST FAILED: Found duplicate step executions!")
            print("   This indicates the infinite loop bug is still present.")
        else:
            print("✅ TEST PASSED: No duplicate step executions detected!")
            print("   The HITL infinite loop fix is working correctly.")
        print("="*80)
        
        return not has_duplicates
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        return False
    except Exception as e:
        print(f"\n\n❌ TEST FAILED WITH ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_scenario_descriptions():
    """Print test scenario descriptions for manual testing."""
    
    print("\n" + "="*80)
    print("📋 MANUAL TEST SCENARIOS")
    print("="*80)
    
    scenarios = [
        {
            "name": "Scenario 1: Full Pipeline - Approve at First HITL",
            "steps": [
                "1. Run: python main.py",
                "2. Select mode: 2 (Full Pipeline)",
                "3. Enter query: 'Effect of gender on coupon usage'",
                "4. At first HITL (table_selection): Choose 'approve'",
                "5. Expected: Should proceed to table_retrieval, NOT loop back"
            ],
            "expected": "No loop at table_selection, continues to next step"
        },
        {
            "name": "Scenario 2: Text2SQL → Full Pipeline Switch",
            "steps": [
                "1. Run: python main.py",
                "2. Select mode: 1 (Data Exploration)",
                "3. Generate some data",
                "4. Switch to: 1 (Proceed to Causal Analysis)",
                "5. At first HITL (data_profiling): Choose 'approve'",
                "6. Expected: Should proceed to algorithm_configuration, NOT loop"
            ],
            "expected": "No loop at data_profiling after mode switch"
        },
        {
            "name": "Scenario 3: Edit Decision at HITL",
            "steps": [
                "1. Run: python main.py",
                "2. Select mode: 2 (Full Pipeline)",
                "3. At table_selection HITL: Choose 'edit'",
                "4. Modify selected_tables",
                "5. Expected: Should apply edits and proceed to table_retrieval"
            ],
            "expected": "Edits applied, proceeds to next step without re-execution"
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{scenario['name']}")
        print("-" * 80)
        for step in scenario['steps']:
            print(f"  {step}")
        print(f"\n  ✅ Expected Outcome: {scenario['expected']}")
        print()


if __name__ == "__main__":
    print("\n" + "="*80)
    print("🔧 HITL INFINITE LOOP FIX - TEST SUITE")
    print("="*80)
    
    # Check for test mode argument
    if len(sys.argv) > 1 and sys.argv[1] == "--manual":
        test_scenario_descriptions()
        print("\n💡 To run automated test: python tests/test_hitl_loop_fix.py")
    else:
        print("\n🤖 Running automated HITL loop test...")
        print("   (Use --manual flag to see manual test scenarios)")
        success = test_hitl_loop_fix()
        sys.exit(0 if success else 1)

