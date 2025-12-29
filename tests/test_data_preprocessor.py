"""Test Data Preprocessor Agent with various data types.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict, Any
from datetime import datetime

from agents.data_explorer.data_preprocessor.agent import DataPreprocessorAgent
from core.state import create_initial_state
from utils.synthetic_data import generate_er_synthetic


def create_test_dataframe_with_mixed_types() -> pd.DataFrame:
    """Create a test DataFrame with mixed data types."""
    np.random.seed(42)
    n_samples = 200
    
    # Continuous variables
    continuous1 = np.random.normal(0, 1, n_samples)
    continuous2 = np.random.uniform(0, 100, n_samples)
    
    # Binary variables
    binary1 = np.random.choice([0, 1], n_samples)
    binary2 = np.random.choice(['Yes', 'No'], n_samples)
    
    # Ordinal variables (low cardinality numeric)
    ordinal1 = np.random.choice([1, 2, 3, 4, 5], n_samples)
    
    # Nominal variables (low cardinality)
    nominal1 = np.random.choice(['A', 'B', 'C'], n_samples)
    nominal2 = np.random.choice(['Low', 'Medium', 'High'], n_samples)
    
    # High cardinality categorical
    high_card = [f"category_{i}" for i in np.random.choice(range(60), n_samples)]
    
    # Variables with nulls
    with_nulls = np.random.normal(0, 1, n_samples)
    null_indices = np.random.choice(n_samples, size=int(n_samples * 0.3), replace=False)
    with_nulls[null_indices] = np.nan
    
    # High null ratio column (will be dropped)
    high_null = np.random.normal(0, 1, n_samples)
    high_null_indices = np.random.choice(n_samples, size=int(n_samples * 0.98), replace=False)
    high_null[high_null_indices] = np.nan
    
    df = pd.DataFrame({
        'continuous1': continuous1,
        'continuous2': continuous2,
        'binary_numeric': binary1,
        'binary_categorical': binary2,
        'ordinal': ordinal1,
        'nominal1': nominal1,
        'nominal2': nominal2,
        'high_cardinality': high_card,
        'with_nulls': with_nulls,
        'high_null_col': high_null,
    })
    
    return df


def test_preprocessor_full_pipeline():
    """Test full preprocessing pipeline."""
    print("\n" + "="*60)
    print("Testing Data Preprocessor - Full Pipeline")
    print("="*60)
    
    # Create test data
    df = create_test_dataframe_with_mixed_types()
    print(f"\n✓ Created test DataFrame: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Dtypes: {df.dtypes.to_dict()}")
    
    # Store in Redis for fetch step
    from utils.redis_df import save_df_parquet
    session_id = f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    redis_key = f"test_db:raw_df:{session_id}"
    save_df_parquet(redis_key, df)
    print(f"✓ Saved to Redis: {redis_key}")
    
    # Initialize agent
    agent = DataPreprocessorAgent(name="test_preprocessor")
    
    # Create initial state
    state = create_initial_state("Test preprocessing", "test_db")
    state["df_redis_key"] = redis_key
    state["session_id"] = session_id
    state["high_cardinality_threshold"] = 50
    state["clean_nulls_ratio"] = 0.95
    state["one_hot_threshold"] = 20
    state["interactive"] = False  # Skip HITL for testing
    
    # Test: Full pipeline
    print("\n--- Running Full Pipeline ---")
    state["current_substep"] = "full_pipeline"
    state = agent.step(state)
    
    if state.get("error"):
        print(f"❌ Error: {state['error']}")
        return False
    
    print("✓ Full pipeline completed")
    
    # Verify results
    print("\n--- Verification ---")
    
    # 1. Check schema detection
    schema = state.get("variable_schema", {})
    if not schema:
        print("❌ Schema not detected")
        return False
    print(f"✓ Schema detected: {len(schema.get('variables', {}))} variables")
    
    # Check mixed data types
    mixed = schema.get("mixed_data_types", False)
    print(f"✓ Mixed data types detected: {mixed}")
    
    stats = schema.get("statistics", {})
    print(f"  - Continuous: {stats.get('n_continuous', 0)}")
    print(f"  - Categorical: {stats.get('n_categorical', 0)}")
    print(f"  - Binary: {stats.get('n_binary', 0)}")
    
    # Check high cardinality detection
    high_card_vars = schema.get("high_cardinality_vars", [])
    print(f"✓ High cardinality vars: {high_card_vars}")
    
    # 2. Check null cleaning
    dropped_cols = state.get("dropped_null_columns", [])
    print(f"✓ Dropped columns: {dropped_cols}")
    if "high_null_col" not in dropped_cols:
        print("⚠ Warning: high_null_col should have been dropped")
    
    # 3. Check encoding
    encoded_cols = state.get("encoded_columns", [])
    print(f"✓ Encoded columns: {encoded_cols}")
    
    # 4. Check DataFrame shape
    df_shape = state.get("df_shape")
    print(f"✓ Final DataFrame shape: {df_shape}")
    
    # 5. Check agent's internal DataFrame
    if agent.df is not None:
        print(f"✓ Agent cached DataFrame: {agent.df.shape}")
        print(f"  Columns: {list(agent.df.columns)[:10]}...")  # Show first 10
    else:
        print("⚠ Warning: Agent DataFrame not cached")
    
    return True


def test_preprocessor_substeps():
    """Test individual preprocessing substeps."""
    print("\n" + "="*60)
    print("Testing Data Preprocessor - Individual Substeps")
    print("="*60)
    
    # Create test data
    df = create_test_dataframe_with_mixed_types()
    
    # Store in Redis
    from utils.redis_df import save_df_parquet
    session_id = f"test_substeps_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    redis_key = f"test_db:raw_df:{session_id}"
    save_df_parquet(redis_key, df)
    
    # Initialize agent
    agent = DataPreprocessorAgent(name="test_preprocessor")
    
    # Create initial state
    state = create_initial_state("Test preprocessing", "test_db")
    state["df_redis_key"] = redis_key
    state["session_id"] = session_id
    state["interactive"] = False
    
    # Test 1: Fetch
    print("\n--- Step 1: Fetch ---")
    state["current_substep"] = "fetch"
    state = agent.step(state)
    
    if state.get("error"):
        print(f"❌ Fetch error: {state['error']}")
        return False
    
    print(f"✓ Fetch completed: {state.get('df_shape')}")
    print(f"  Redis key: {state.get('df_redis_key')}")
    print(f"  Agent cached: {agent.df is not None}")
    
    # Test 2: Schema Detection
    print("\n--- Step 2: Schema Detection ---")
    state["current_substep"] = "schema_detection"
    state["high_cardinality_threshold"] = 50
    state = agent.step(state)
    
    if state.get("error"):
        print(f"❌ Schema detection error: {state['error']}")
        return False
    
    schema = state.get("variable_schema", {})
    print(f"✓ Schema detected: {len(schema.get('variables', {}))} variables")
    
    # Show sample variable info
    variables = schema.get("variables", {})
    sample_var = list(variables.keys())[0] if variables else None
    if sample_var:
        var_info = variables[sample_var]
        print(f"  Sample variable '{sample_var}':")
        print(f"    - Type: {var_info.get('data_type')}")
        print(f"    - Cardinality: {var_info.get('cardinality')}")
        print(f"    - Missing ratio: {var_info.get('missing_ratio', 0):.2%}")
    
    # Test 3: Clean Nulls
    print("\n--- Step 3: Clean Nulls ---")
    state["current_substep"] = "clean_nulls"
    state["clean_nulls_ratio"] = 0.95
    state = agent.step(state)
    
    if state.get("error"):
        print(f"❌ Clean nulls error: {state['error']}")
        return False
    
    dropped = state.get("dropped_null_columns", [])
    print(f"✓ Clean nulls completed: dropped {len(dropped)} columns")
    print(f"  Dropped: {dropped}")
    
    # Test 4: Encode
    print("\n--- Step 4: Encode ---")
    state["current_substep"] = "encode"
    state["one_hot_threshold"] = 20
    state = agent.step(state)
    
    if state.get("error"):
        print(f"❌ Encode error: {state['error']}")
        return False
    
    encoded = state.get("encoded_columns", [])
    print(f"✓ Encode completed: encoded {len(encoded)} columns")
    print(f"  Encoded: {encoded}")
    print(f"  Final shape: {state.get('df_shape')}")
    
    return True


def test_schema_detection_details():
    """Test schema detection with detailed verification."""
    print("\n" + "="*60)
    print("Testing Schema Detection - Detailed Verification")
    print("="*60)
    
    df = create_test_dataframe_with_mixed_types()
    
    from utils.redis_df import save_df_parquet
    session_id = f"test_schema_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    redis_key = f"test_db:raw_df:{session_id}"
    save_df_parquet(redis_key, df)
    
    agent = DataPreprocessorAgent(name="test_preprocessor")
    state = create_initial_state("Test schema", "test_db")
    state["df_redis_key"] = redis_key
    state["session_id"] = session_id
    state["high_cardinality_threshold"] = 50
    state["interactive"] = False
    
    # Run fetch and schema detection
    state["current_substep"] = "fetch"
    state = agent.step(state)
    
    state["current_substep"] = "schema_detection"
    state = agent.step(state)
    
    if state.get("error"):
        print(f"❌ Error: {state['error']}")
        return False
    
    schema = state.get("variable_schema", {})
    variables = schema.get("variables", {})
    
    print("\n--- Variable Type Verification ---")
    
    # Verify continuous
    continuous_vars = [v for v, info in variables.items() 
                      if info.get("data_type") == "Continuous"]
    print(f"✓ Continuous variables ({len(continuous_vars)}): {continuous_vars}")
    
    # Verify binary
    binary_vars = [v for v, info in variables.items() 
                  if info.get("data_type") == "Binary"]
    print(f"✓ Binary variables ({len(binary_vars)}): {binary_vars}")
    
    # Verify ordinal
    ordinal_vars = [v for v, info in variables.items() 
                   if info.get("data_type") == "Ordinal"]
    print(f"✓ Ordinal variables ({len(ordinal_vars)}): {ordinal_vars}")
    
    # Verify nominal
    nominal_vars = [v for v, info in variables.items() 
                   if info.get("data_type") == "Nominal"]
    print(f"✓ Nominal variables ({len(nominal_vars)}): {nominal_vars}")
    
    # Verify high cardinality detection
    high_card = schema.get("high_cardinality_vars", [])
    print(f"✓ High cardinality variables: {high_card}")
    
    # Verify mixed data types
    mixed = schema.get("mixed_data_types", False)
    print(f"✓ Mixed data types: {mixed}")
    
    # Detailed info for each variable
    print("\n--- Detailed Variable Information ---")
    for var_name, var_info in list(variables.items())[:5]:  # Show first 5
        print(f"\n{var_name}:")
        print(f"  Type: {var_info.get('data_type')}")
        print(f"  Dtype: {var_info.get('dtype')}")
        print(f"  Cardinality: {var_info.get('cardinality')}")
        print(f"  Missing ratio: {var_info.get('missing_ratio', 0):.2%}")
        if var_info.get('unique_values'):
            unique_vals = var_info.get('unique_values', [])
            if len(unique_vals) <= 10:
                print(f"  Unique values: {unique_vals}")
    
    return True


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("Data Preprocessor Agent Tests")
    print("="*60)
    
    results = []
    
    try:
        # Test 1: Full pipeline
        result1 = test_preprocessor_full_pipeline()
        results.append(("Full Pipeline", result1))
        
        # Test 2: Individual substeps
        result2 = test_preprocessor_substeps()
        results.append(("Individual Substeps", result2))
        
        # Test 3: Schema detection details
        result3 = test_schema_detection_details()
        results.append(("Schema Detection Details", result3))
        
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    all_passed = True
    for test_name, result in results:
        status = "✓ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if not result:
            all_passed = False
    
    if all_passed:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

