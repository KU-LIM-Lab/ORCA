"""
Script for ATE generation
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
    import sys
    from pathlib import Path
    current_dir = Path(__file__).parent
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
    try:
        from ate_calculator import calculate_ate
        from reef_data_loader import REEFDataLoader
    except ImportError:
        project_root = current_dir.parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        from REEF.generate_ate.ate_calculator import calculate_ate
        from REEF.generate_ate.reef_data_loader import REEFDataLoader


def load_queries_from_yaml(yaml_path: str) -> List[Dict[str, Any]]:
    """
    Load queries from YAML file
    
    Args:
        yaml_path: YAML file path
    
    Returns:
        Dict list of queries
    """
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    return data.get('queries', [])


def generate_question(treatment: str, outcome: str) -> str:
    """
    generatr question from treatment and outcome
    
    Args:
        treatment: treatment variable name
        outcome: outcome variable name
    
    Returns:
        question
    """
    treatment_clean = treatment.split('.')[-1] if '.' in treatment else treatment
    outcome_clean = outcome.split('.')[-1] if '.' in outcome else outcome
    
    return f"What is the causal effect of {treatment_clean} on {outcome_clean}?"


def resolve_variable_name(
    var_name: str,
    df: pd.DataFrame,
    table_prefix: Optional[str] = None
) -> str:
    """    
    Args:
        var_name: variable name (ex: "unit_price" or "order_items.unit_price")
        df: dataframe
        table_prefix: table (ex: "order_items")
    
    Returns:
        column name
    """
    if '.' in var_name:
        parts = var_name.split('.')
        if len(parts) == 2:
            table, col = parts
            if f"{table}.{col}" in df.columns:
                return f"{table}.{col}"
            elif col in df.columns:
                return col
    
    if table_prefix:
        full_name = f"{table_prefix}.{var_name}"
        if full_name in df.columns:
            return full_name
    
    if var_name in df.columns:
        return var_name
    
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
    Calcuate ATE for single query
    
    Args:
        query_config: query configuration dictionary
        loader: REEF data loader
        output_dir: output directory (if None, don't save)
        verbose: print details
    
    Returns:
        output dict
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
    estimator = query_config.get('estimator') 
    
    if verbose:
        print(f"\nProcessing: {treatment} -> {outcome}")
    
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
            "n_samples": result["n_samples"],
            "sql_query" : sql_query
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
    Calcuate ATE for multiple query
    
    Args:
        query_config: query configuration dictionary
        loader: REEF data loader
        output_dir: output directory (if None, don't save)
        verbose: print details
    
    Returns:
        output dict
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
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        if verbose:
            print(f"\nResults saved to {output_path}")
    
    if verbose:
        successful = sum(1 for r in results if "error" not in r)
        failed = len(results) - successful
        print(f"\nSummary: {successful} successful, {failed} failed")
    
    return results

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_name", default="reef_db")
    parser.add_argument("--queries", default="REEF/generate_ate/ate_queries.yaml")
    parser.add_argument("--output", default="REEF/generate_ate/causal_analysis.json")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    queries = load_queries_from_yaml(args.queries)

    results = generate_ate_data(
        queries=queries,
        db_name=args.db_name,
        output_path=args.output,
        verbose=not args.quiet
    )
    return 0 if all("error" not in r for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())

