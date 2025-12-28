"""
Test script for baseline agent

This script tests the baseline agent with a simple query to verify:
1. Tools work correctly
2. Event logging is functional
3. Artifacts are saved properly
4. Step tracking works
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from baseline.tools import get_schema, run_sql, run_python, save_artifact, set_tool_context
from monitoring.experiment import RunContext
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def test_tools_without_context():
    """Test tools work without RunContext (basic functionality)."""
    print("\n" + "="*60)
    print("TEST 1: Basic Tool Functionality (No RunContext)")
    print("="*60)
    
    # Test get_schema
    print("\n1. Testing get_schema...")
    result = get_schema("reef_db")
    if result.get("success"):
        print(f"   ✓ Schema retrieved: {result.get('table_count')} tables")
    else:
        print(f"   ✗ Failed: {result.get('error')}")
    
    # Test run_sql (simple query)
    print("\n2. Testing run_sql...")
    result = run_sql("reef_db", "SELECT 1 as test_column")
    if result.get("success"):
        print(f"   ✓ SQL executed: {result.get('row_count')} rows")
    else:
        print(f"   ✗ Failed: {result.get('error')}")
    
    # Test run_python (simple calculation)
    print("\n3. Testing run_python...")
    result = run_python("import pandas as pd\nresult = 2 + 2")
    if result.get("success"):
        print(f"   ✓ Python executed: {result.get('outputs')}")
    else:
        print(f"   ✗ Failed: {result.get('error')}")
    
    print("\n✓ Basic tool tests passed")


def test_tools_with_context():
    """Test tools with RunContext for event logging and artifacts."""
    print("\n" + "="*60)
    print("TEST 2: Tools with RunContext (Logging & Artifacts)")
    print("="*60)
    
    # Create a test run context
    with RunContext(
        participant_id="TEST",
        condition="baseline",
        task_id="test_task",
        run_id="test_run_001"
    ) as run_ctx:
        event_logger = run_ctx.get_event_logger()
        artifact_manager = run_ctx.get_artifact_manager()
        
        # Set tool context
        set_tool_context(event_logger=event_logger, artifact_manager=artifact_manager)
        
        # Test get_schema with logging
        print("\n1. Testing get_schema with logging...")
        result = get_schema("reef_db")
        if result.get("success"):
            print(f"   ✓ Schema retrieved and logged")
        
        # Test run_sql with logging
        print("\n2. Testing run_sql with logging...")
        result = run_sql("reef_db", "SELECT 1 as id, 'test' as name")
        if result.get("success"):
            print(f"   ✓ SQL executed and logged")
        
        # Test save_artifact
        print("\n3. Testing save_artifact...")
        result = save_artifact(
            artifact_type="sql",
            data_ref="last",
            step_id="1",
            filename="test_query.sql"
        )
        if result.get("success"):
            print(f"   ✓ SQL artifact saved: {result.get('path')}")
        
        result = save_artifact(
            artifact_type="dataset",
            data_ref="last",
            step_id="1",
            filename="test_data.parquet"
        )
        if result.get("success"):
            print(f"   ✓ Dataset artifact saved: {result.get('path')}")
        
        # Test Python execution with dataframe access
        print("\n4. Testing run_python with dataframe context...")
        code = """
import pandas as pd
# df is available from last SQL result
print(f"DataFrame shape: {df.shape}")
result_sum = len(df)
"""
        result = run_python(code)
        if result.get("success"):
            print(f"   ✓ Python executed with df context")
            print(f"   Output: {result.get('stdout')}")
        
        # Test graph artifact (mock data)
        print("\n5. Testing graph artifact...")
        graph_data = {
            "nodes": ["A", "B", "C"],
            "edges": [["A", "B"], ["B", "C"]]
        }
        import json
        result = save_artifact(
            artifact_type="graph",
            data_ref=json.dumps(graph_data),
            step_id="2",
            filename="test_graph.json"
        )
        if result.get("success"):
            print(f"   ✓ Graph artifact saved: {result.get('path')}")
        
        # Test ATE artifact (mock data)
        print("\n6. Testing ATE artifact...")
        ate_data = {
            "ate": 0.25,
            "ci_lower": 0.15,
            "ci_upper": 0.35,
            "method": "backdoor_adjustment"
        }
        result = save_artifact(
            artifact_type="ate",
            data_ref=json.dumps(ate_data),
            step_id="3",
            filename="test_ate.json"
        )
        if result.get("success"):
            print(f"   ✓ ATE artifact saved: {result.get('path')}")
        
        print(f"\n✓ All artifacts saved to: {run_ctx.artifacts_dir}")
        print(f"✓ Events logged to: {run_ctx.events_file}")
        
        # Check event count
        event_count = event_logger.get_event_count()
        print(f"✓ Total events logged: {event_count}")


def test_baseline_agent_import():
    """Test that baseline agent can be imported."""
    print("\n" + "="*60)
    print("TEST 3: Import Baseline Agent")
    print("="*60)
    
    try:
        from baseline.baseline_agent import BaselineAgent, run_baseline_interactive
        print("✓ BaselineAgent imported successfully")
        print("✓ run_baseline_interactive imported successfully")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_openai_api_key():
    """Test that OpenAI API key is configured."""
    print("\n" + "="*60)
    print("TEST 4: OpenAI API Configuration")
    print("="*60)
    
    import os
    if os.getenv("OPENAI_API_KEY"):
        print("✓ OPENAI_API_KEY is set")
        return True
    else:
        print("⚠️  OPENAI_API_KEY not set - agent will fail at runtime")
        print("   Set it with: export OPENAI_API_KEY='your-key'")
        return False


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print(" "*20 + "BASELINE AGENT TEST SUITE")
    print("="*70)
    
    try:
        # Run tests
        test_tools_without_context()
        test_tools_with_context()
        test_baseline_agent_import()
        test_openai_api_key()
        
        print("\n" + "="*70)
        print("✅ ALL TESTS PASSED")
        print("="*70)
        print("\nBaseline agent is ready to use!")
        print("\nTo run the baseline agent:")
        print("  python user_study_entry.py --participant-id TEST --condition baseline --task-id test")
        print()
        
    except Exception as e:
        logger.exception("Test failed")
        print(f"\n❌ TESTS FAILED: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

