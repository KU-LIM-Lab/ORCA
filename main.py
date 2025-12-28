import logging
import sys
import os
import warnings
from datetime import datetime
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from typing import Optional, Dict, Any
import argparse

import pandas as pd

from core.state import create_initial_state
from orchestration.graph import create_orchestration_graph
from utils.system_init import initialize_system
from utils.synthetic_data import generate_er_synthetic
from utils.llm import get_llm
from agents.data_explorer.table_explorer.agent import TableExplorerAgent
from agents.data_explorer.table_recommender.agent import TableRecommenderAgent
from agents.data_explorer.text2sql_generator.agent import Text2SQLGeneratorAgent
from agents.data_explorer.agent import DataExplorerAgent


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def select_analysis_mode() -> str:
    """Prompt user to select analysis mode at session start."""
    print("\n" + "="*60)
    print("🤖 ORCA: Welcome! Select analysis mode:")
    print("="*60)
    print("  1) Data Exploration")
    print("     - Explore database tables and extract data")
    print("     - Useful for understanding the data and its relationships")
    print()
    print("  2) Fully Automated Causal Analysis Pipeline")
    print("     - Complete end-to-end causal analysis")
    print("     - Includes data exploration, causal discovery, and inference")
    print("="*60)
    
    while True:
        choice = input("\n💬 Your choice (1 or 2): ").strip()
        if choice == "1":
            return "data_exploration"
        elif choice == "2":
            return "full_pipeline"
        else:
            print("❌ Invalid choice. Please enter 1 or 2.")


def prompt_next_step_after_data() -> str:
    """
    Ask user what to do after data is generated/extracted.
    Returns: '1' (causal analysis), '2' (continue exploration), '3' (exit)
    """
    print("\n" + "="*60)
    print("📌 Data Ready! What's Next?")
    print("="*60)
    print("Your data has been successfully extracted and is ready to use.")
    print()
    print("Options:")
    print("  1) Proceed to Causal Analysis with this data")
    print("  2) Continue Data Exploration")
    print("  3) Exit session")
    print("="*60)
    
    while True:
        choice = input("\n💬 Your choice (1-3): ").strip().lower()
        
        if choice in ["quit", "exit", "q"]:
            if confirm_quit():
                return "3"
            else:
                continue
        
        if choice in ["1", "2", "3"]:
            return choice
        else:
            print("❌ Invalid choice. Please enter 1, 2, or 3.")


def confirm_quit() -> bool:
    """
    Ask user to confirm they want to quit the session.
    Returns True if user confirms quit, False otherwise.
    """
    print("\n" + "="*60)
    print("⚠️  WARNING: Exiting will clear all session data!")
    print("="*60)
    print("   - All exploration results will be lost")
    print("   - Any preprocessed data will be deleted")
    print("   - You will need to start over")
    print()
    confirm = input("❓ Are you sure you want to quit? (yes/no): ").strip().lower()
    if confirm in ["yes", "y"]:
        return True
    else:
        print("✓ Continuing session...")
        return False


def select_data_exploration_mode() -> str:
    """Prompt user to select data exploration method."""
    print("\n" + "="*60)
    print("📊 Data Exploration Mode")
    print("="*60)
    print("Select exploration method:")
    print()
    print("  1) Table Explorer")
    print("     - Explore specific table schema and relationships")
    print()
    print("  2) Table Recommender")
    print("     - Get table recommendations from query/document")
    print()
    print("  3) Text2SQL Generator")
    print("     - Generate SQL from natural language query")
    print()
    print("  4) Automated Data Pipeline")
    print("     - Automatically select relevant tables, generate SQL, and extract data")
    print("     - Data is ready for immediate use in causal discovery/inference")
    print()
    print("  Or type 'done' to switch to Causal Analysis mode")
    print("="*60)
    print("\n💡 Tips:")
    print("   • Type 'done' to try a different exploration method")
    print("   • Type 'quit' to exit the entire session")
    
    while True:
        choice = input("\n💬 Your choice (1-4 or 'done'): ").strip().lower()
        
        if choice == "done":
            return "done"
        elif choice == "1":
            return "table_explorer"
        elif choice == "2":
            return "table_recommender"
        elif choice == "3":
            return "text2sql"
        elif choice == "4":
            return "full_pipeline"
        elif choice in ["quit", "exit", "q"]:
            if confirm_quit():
                return "done"
            else:
                continue
        else:
            print("❌ Invalid choice. Please enter 1-4 or 'done'.")


def prompt_data_reuse(previous_state: Dict[str, Any]) -> str:
    """Prompt user whether to reuse existing data or explore new data."""
    print("\n" + "="*60)
    print("ℹ️  Previous data available:")
    print("="*60)
    
    # Display previous data info
    if previous_state.get("exploration_mode"):
        print(f"   Exploration method: {previous_state['exploration_mode']}")
    if previous_state.get("selected_tables"):
        tables = previous_state["selected_tables"]
        if isinstance(tables, list):
            print(f"   Tables: {', '.join(tables)}")
        else:
            print(f"   Tables: {tables}")
    if previous_state.get("sql_query") or previous_state.get("final_sql"):
        sql = previous_state.get("final_sql") or previous_state.get("sql_query")
        sql_preview = sql[:100] + "..." if len(sql) > 100 else sql
        print(f"   SQL: {sql_preview}")
    if previous_state.get("df_shape"):
        shape = previous_state["df_shape"]
        print(f"   Data shape: {shape[0]} rows × {shape[1]} columns")
    
    # Show preprocessing status
    if previous_state.get("data_preprocessing_completed"):
        print(f"   Status: ✓ Preprocessed and ready for analysis")
    else:
        print(f"   Status: ⚠️  Raw data (will be preprocessed if used for causal analysis)")
    
    print()
    print("Options:")
    print("  1) Use this data for causal analysis")
    print("  2) Explore new data (clear previous data)")
    print("  3) Switch to data exploration mode")
    print("  4) Exit session")
    print("="*60)
    
    while True:
        choice = input("\n💬 Your choice (1-4): ").strip().lower()
        
        # Handle quit commands
        if choice in ["quit", "exit", "q", "4"]:
            if confirm_quit():
                return "4"  # Exit
            else:
                continue
        
        if choice in ["1", "2", "3"]:
            return choice
        else:
            print("❌ Invalid choice. Please enter 1, 2, or 3.")


def run_data_exploration_only(
    query: str,
    db_id: str,
    session_id: str,
    exploration_mode: str,
    llm: Any,
) -> Dict[str, Any]:
    """
    Run data exploration only.
    
    Args:
        query: User query or table name
        db_id: Database identifier
        session_id: Session identifier
        exploration_mode: One of 'table_explorer', 'table_recommender', 'text2sql', 'full_pipeline'
        llm: LLM instance
    
    Returns:
        Dictionary with exploration results and state
    """
    try:
        # Initialize system
        init = initialize_system(db_id, "postgresql", {})
        if not (init and init.is_connected):
            raise RuntimeError("System initialization failed")
        
        logger.info(f"Running data exploration mode: {exploration_mode}")
        
        result_state = {}
        
        if exploration_mode == "table_explorer":
            print(f"\n🔍 Analyzing table structure: {query}")
            print("   Please wait while we examine the table schema, relationships, and data characteristics...")
            agent = TableExplorerAgent(llm=llm, name="table_explorer")
            agent_state = {
                "db_id": db_id,
                "input": query
            }
            result = agent.execute(agent_state)
            
            if result.success and result.data:
                result_state = result.data
                result_state["exploration_mode"] = exploration_mode
                # Don't set data_exploration_status to "completed" for table_explorer
                print(f"\n✅ Table exploration completed!")
                if result.data.get("final_output"):
                    from utils.prettify import print_final_output_explorer
                    prettified = print_final_output_explorer(result.data['final_output'])
                    print(f"\n{prettified}")
            else:
                raise RuntimeError(f"Table exploration failed: {result.error}")
        
        elif exploration_mode == "table_recommender":
            print(f"\n🔍 Getting table recommendations for: {query}")
            agent = TableRecommenderAgent(llm=llm, name="table_recommender")
            agent_state = {
                "db_id": db_id,
                "input": query,
                "input_type": "text"
            }
            result = agent.execute(agent_state)
            
            if result.success and result.data:
                result_state = result.data
                result_state["exploration_mode"] = exploration_mode
                # Don't set data_exploration_status to "completed" for table_recommender
                result_state["selected_tables"] = result.data.get("recommended_tables", [])
                print(f"\n✅ Table recommendation completed!")
                if result.data.get("final_output"):
                    from utils.prettify import print_final_output_recommender
                    prettified = print_final_output_recommender(result.data['final_output'])
                    print(f"\n{prettified}")
                elif result.data.get("recommended_tables"):
                    print(f"   Recommended tables: {', '.join(result.data['recommended_tables'])}")
            else:
                raise RuntimeError(f"Table recommendation failed: {result.error}")
        
        elif exploration_mode == "text2sql":
            print(f"\n🔍 Generating SQL query for: {query}")
            print("   Please wait while we convert your natural language query into SQL...")
            agent = Text2SQLGeneratorAgent(llm=llm, name="text2sql_generator")
            agent_state = {
                "db_id": db_id,
                "query": query,
                "messages": [],
                "evidence": "",
                "analysis_mode": "data_exploration"
            }
            result = agent.execute(agent_state)
            
            if result.success and result.data:
                result_state = result.data
                result_state["exploration_mode"] = exploration_mode
                result_state["data_exploration_status"] = "completed"
                result_state["sql_query"] = result.data.get("final_sql", "")
                result_state["final_sql"] = result.data.get("final_sql", "")
                
                # Execute SQL and get data (RAW - not preprocessed)
                if result.data.get("result"):
                    from utils.redis_df import save_df_parquet
                    import pandas as pd
                    
                    # Convert result to DataFrame
                    df = pd.DataFrame(result.data["result"])
                    if result.data.get("columns"):
                        df.columns = result.data["columns"]
                    
                    # Save RAW data to Redis (will be preprocessed later if needed)
                    df_key = f"{db_id}:df_raw:{session_id}"
                    save_df_parquet(df_key, df)
                    result_state["df_raw_redis_key"] = df_key
                    result_state["df_shape"] = df.shape
                    result_state["columns"] = list(df.columns)
                    
                    # Mark that preprocessing is NOT done yet
                    result_state["data_preprocessing_completed"] = False
                    
                    print(f"\n✅ SQL generation and execution completed!")
                    
                    from utils.prettify import print_final_output_sql
                    output_dict = {
                        "sql": result.data.get('final_sql', ''),
                        "result": result.data.get("result", []),
                        "columns": result.data.get("columns", []),
                        "error": result.data.get("error"),
                        "llm_review": result.data.get("llm_review")
                    }
                    prettified = print_final_output_sql(output_dict)
                    print(f"\n{prettified}")
                    print(f"\n   Data shape: {df.shape[0]} rows × {df.shape[1]} columns")
                    print(f"   ⚠️  Note: Data is raw (not preprocessed)")
            else:
                raise RuntimeError(f"Text2SQL generation failed: {result.error}")
        
        elif exploration_mode == "full_pipeline":
            print(f"\n🔍 Running full data exploration pipeline for: {query}")
            print("   Please wait while we select tables, generate SQL, and extract data...")
            agent = DataExplorerAgent(llm=llm, name="data_explorer")
            agent_state = {
                "db_id": db_id,
                "initial_query": query,
                "input": query,
                "current_substep": "full_pipeline",
                "session_id": session_id,
                "persist_to_redis": True,
                "analysis_mode": "data_exploration"
            }
            
            result_state = agent.step(agent_state)
            result_state["exploration_mode"] = exploration_mode
            
            if not result_state.get("error"):
                result_state["data_exploration_status"] = "completed"
                # Check if preprocessing was completed
                if result_state.get("data_preprocessing_completed"):
                    result_state["data_preprocessing_completed"] = True
                
                print(f"\n✅ Data exploration pipeline completed!")
                
                if result_state.get("selected_tables"):
                    print(f"   Tables: {', '.join(result_state['selected_tables'])}")
                if result_state.get("sql_query") or result_state.get("final_sql"):
                    sql = result_state.get("final_sql") or result_state.get("sql_query")
                    print(f"   SQL: {sql[:100]}...")
                if result_state.get("df_shape"):
                    shape = result_state["df_shape"]
                    print(f"   Data shape: {shape[0]} rows × {shape[1]} columns")
                if result_state.get("data_preprocessing_completed"):
                    print(f"   ✓ Data preprocessing completed")
                
                # Show next steps
                print("\n" + "="*60)
                print("📌 What's Next?")
                print("="*60)
                print("You can now:")
                print("  • Continue exploring")
                print("  • Change mode: Type 'done' to select different analysis mode")
                print("  • Exit session: Type 'quit' to end")
                print("="*60)
            else:
                raise RuntimeError(f"Data exploration pipeline failed: {result_state.get('error')}")
        
        return {
            "success": True,
            "state": result_state
        }
    
    except Exception as e:
        logger.exception(f"Data exploration failed: {e}")
        return {
            "success": False,
            "state": {"error": str(e)}
        }


def print_previous_state_summary(previous_state: Dict[str, Any]) -> None:
    """Print a summary of the previous state."""
    if not previous_state:
        return
    
    print("\n📋 Session State:")
    if previous_state.get("exploration_mode"):
        print(f"   Mode: {previous_state['exploration_mode']}")
    if previous_state.get("data_exploration_status") == "completed":
        print(f"   Status: Data exploration completed ✓")
        if previous_state.get("df_shape"):
            shape = previous_state["df_shape"]
            print(f"   Data: {shape[0]} rows × {shape[1]} columns")


def run_full_pipeline(
    query: str,
    db_id: str = "reef_db",
    session_id: Optional[str] = None,
    previous_state: Optional[Dict[str, Any]] = None,
    planner_config: Optional[Dict[str, Any]] = None,
    executor_config: Optional[Dict[str, Any]] = None,
    orchestration_config: Optional[Dict[str, Any]] = None,
    use_synthetic_df: bool = False,
    analysis_mode: str = "full_pipeline",
    event_logger: Optional[Any] = None,
) -> Dict[str, Any]:
    # 1) Initialize system (db + metadata)
    init = initialize_system(db_id, "postgresql", {})
    if not (init and init.is_connected):
        raise RuntimeError("System initialization failed")

    # 2) Build orchestration graph
    graph = create_orchestration_graph(
        planner_config=planner_config,
        executor_config=executor_config,
        orchestration_config=orchestration_config,
        metrics_collector=None,
        event_logger=event_logger,
    )
    graph.compile()

    # 3) Create initial state with session_id
    state = create_initial_state(query, db_id, session_id=session_id)
    state["analysis_mode"] = analysis_mode
    if use_synthetic_df:
        df, _meta = generate_er_synthetic(n_nodes=5, edge_prob=0.3, n_samples=300, seed=123)
        try:
            from utils.redis_client import redis_client
            key = f"{db_id}:df_preprocessed"
            redis_client.set(key, df.to_json(orient="split"))
            state["df_preprocessed_key"] = key
        except Exception:
            warnings.warn("Failed to save dataframe to Redis", stacklevel=2)
            state["df_preprocessed"] = df

    # Optional seeds to pass planner gating for data exploration
    state.setdefault("schema_info", {"tables": []})
    state.setdefault("table_metadata", {})

    # 4) Prepare context from previous state if available
    context = {}
    if previous_state:
        # If data exploration was completed, reuse the data
        if previous_state.get("data_exploration_status") == "completed":
            # Determine which steps to skip based on what was completed
            skip_steps = []
            
            # Always skip table selection and retrieval if data exists
            skip_steps.extend(["table_selection", "table_retrieval"])
            
            # Only skip preprocessing if it was actually completed
            if previous_state.get("data_preprocessing_completed"):
                skip_steps.append("data_preprocessing")
                # Use preprocessed data
                if previous_state.get("df_redis_key"):
                    context["df_redis_key"] = previous_state["df_redis_key"]
                logger.info("Reusing preprocessed data from previous query")
            else:
                # Data exists but not preprocessed - need to preprocess before causal discovery
                if previous_state.get("df_raw_redis_key"):
                    context["df_raw_redis_key"] = previous_state["df_raw_redis_key"]
                elif previous_state.get("df_redis_key"):
                    # Legacy: if only df_redis_key exists, assume it's raw
                    context["df_raw_redis_key"] = previous_state["df_redis_key"]
                logger.info("Raw data available - will preprocess before causal analysis")
            
            # Set skip steps
            context["skip"] = skip_steps
            
            # Preserve SQL query
            if previous_state.get("final_sql"):
                context["final_sql"] = previous_state["final_sql"]
            elif previous_state.get("sql_query"):
                context["sql_query"] = previous_state["sql_query"]
            
            context["data_exploration_status"] = "completed"
            
            # Preserve exploration metadata
            if previous_state.get("exploration_mode"):
                context["exploration_mode"] = previous_state["exploration_mode"]
            if previous_state.get("selected_tables"):
                context["selected_tables"] = previous_state["selected_tables"]
            if previous_state.get("df_shape"):
                context["df_shape"] = previous_state["df_shape"]
            if previous_state.get("columns"):
                context["columns"] = previous_state["columns"]
            if previous_state.get("data_preprocessing_completed") is not None:
                context["data_preprocessing_completed"] = previous_state["data_preprocessing_completed"]

    # 5) Execute
    result_state = graph.execute(query, context={**state, **context}, session_id=session_id)

    logger.info("Selected algorithms: %s", result_state.get("selected_algorithms"))
    logger.info("Selected graph edges: %d", len(result_state.get("selected_graph", {}).get("edges", [])))
    if result_state.get("final_report"):
        logger.info("Report sections: %s", list(result_state["final_report"].get("sections", {}).keys()))

    return {
        "success": not bool(result_state.get("error")),
        "state": result_state,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ORCA Agent entrypoint")
    parser.add_argument("--query", help="Query to execute (if not provided, reads from terminal)")
    parser.add_argument("--db-id", default="reef_db", help="Database ID")
    parser.add_argument("--interactive", action="store_true", help="Enable interactive (HITL) mode")
    parser.add_argument("--print-report", action="store_true", help="Print final report markdown")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG","INFO","WARNING","ERROR","CRITICAL"], help="Log level")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # If query provided as argument, run once and exit
    if args.query:
        planner_cfg: Dict[str, Any] = {}
        executor_cfg: Dict[str, Any] = {}
        orchestration_cfg: Dict[str, Any] = {"interactive": bool(args.interactive)}
        
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        output = run_full_pipeline(
            args.query,
            db_id=args.db_id,
            session_id=session_id,
            planner_config=planner_cfg,
            executor_config=executor_cfg,
            orchestration_config=orchestration_cfg,
            use_synthetic_df=False,
        )
        if output["success"]:
            logger.info("Success")
            if args.print_report and output["state"].get("final_report"):
                fr = output["state"].get("final_report", {})
                print("\n=== Final Report (Markdown) ===")
                print(fr.get("markdown", ""))
        else:
            logger.error("Failed: %s", output["state"].get("error"))
            sys.exit(1)
    else:
        # Interactive mode: loop until user exits
        print("\n" + "="*60)
        print("🤖 ORCA: Causal Analysis System")
        print("="*60)
        print("\n💡 Navigation Tips:")
        print("   • Type 'done' to return to change modes")
        print("   • Type 'quit' to exit the entire session")
        print("   • After data exploration, you can reuse data for subsequent queries")
        print()
        
        # Initialize session
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        previous_state: Optional[Dict[str, Any]] = None
        
        planner_cfg: Dict[str, Any] = {}
        executor_cfg: Dict[str, Any] = {}
        orchestration_cfg: Dict[str, Any] = {"interactive": bool(args.interactive)}
        
        # Initialize LLM for data exploration
        llm = get_llm(model="gpt-4o-mini", temperature=0.3, provider="openai")
        
        # Step 1: Select analysis mode at session start
        analysis_mode = select_analysis_mode()
        
        # Main query loop
        while True:
            print("\n" + "-"*60)
            
            # Step 2: Execute based on analysis mode
            if analysis_mode == "data_exploration":
                # For data exploration, ask for sub-mode selection
                exploration_mode = select_data_exploration_mode()
                
                # Handle 'done' - user wants to change analysis mode
                if exploration_mode == "done":
                    print("✓ Returning to analysis mode selection...")
                    analysis_mode = select_analysis_mode()
                    continue  # Start over with new analysis mode
                
                # Each exploration mode has its own interaction loop
                if exploration_mode == "table_explorer":
                    # Show available tables once at the start
                    try:
                        from utils.redis_client import redis_client
                        import json
                        
                        # Try to get table list from table_relations (from runner.py)
                        relations_key = f"{args.db_id}:table_relations"
                        relations_raw = redis_client.get(relations_key)
                        
                        available_tables = []
                        if relations_raw:
                            relations_info = json.loads(relations_raw)
                            source_schema = relations_info.get("source_schema", {})
                            available_tables = list(source_schema.keys())
                        
                        if available_tables:
                            print(f"\n📋 Available tables in {args.db_id}:")
                            for i, table in enumerate(sorted(available_tables), 1):
                                print(f"   {i}. {table}")
                            print()
                        else:
                            print(f"\n⚠️  No tables found in database '{args.db_id}'")
                            print("💡 Tip: Run 'python utils/data_prep/runner.py' to generate metadata")
                    except Exception as e:
                        print(f"\n⚠️  Could not retrieve table list: {e}")
                        logger.exception("Failed to retrieve table list")
                    
                    exit_program = False
                    while True:
                        query = input("\n🧑 Enter table name to explore (or 'done' to select another exploration method): ").strip()
                        
                        if query.lower() == "done":
                            print("✓ Returning to exploration method selection...")
                            break  # Exit table explorer loop, return to select_data_exploration_mode
                        
                        # Handle empty input - stay in current mode
                        if not query:
                            continue
                        
                        # Handle 'quit' - exit program
                        if query.lower() in ["exit", "quit", "q"]:
                            if confirm_quit():
                                print("👋 Goodbye!")
                                exit_program = True
                                break  # Exit table explorer loop and program
                            else:
                                continue
                        
                        try:
                            output = run_data_exploration_only(
                                query=query,
                                db_id=args.db_id,
                                session_id=session_id,
                                exploration_mode=exploration_mode,
                                llm=llm,
                            )
                            
                            if output["success"]:
                                pass
                            else:
                                print(f"❌ Error: {output['state'].get('error')}")
                                print("💡 Tip: Check the available tables listed above")
                        
                        except Exception as e:
                            logger.exception("Table exploration failed")
                            print(f"❌ Error: {e}")
                    
                    # If user chose to exit, break main loop
                    if exit_program:
                        break
                    
                    # After table explorer loop, continue to next iteration
                    continue
                
                elif exploration_mode == "table_recommender":
                    query = input("\n🧑 Enter your analysis objective (or 'done' for another exploration method): ").strip()
                elif exploration_mode == "text2sql":
                    query = input("\n🧑 Enter your data query in natural language (or 'done' for another exploration method): ").strip()
                else:  # full_pipeline
                    query = input("\n🧑 What would you like to analyze? (e.g., 'effect of gender on purchase', or 'done' for another method): ").strip()
                
                # Handle 'done' - return to exploration method selection
                if query.lower() == "done":
                    print("✓ Returning to exploration method selection...")
                    continue  # Go back to select_data_exploration_mode
                
                # Handle empty input - stay in current mode
                if not query:
                    continue
                
                # Handle 'quit' - exit program
                if query.lower() in ["exit", "quit", "q"]:
                    if confirm_quit():
                        print("👋 Goodbye!")
                        break  # Exit main loop
                    else:
                        continue  # Stay in current mode
                
                # Execute non-table-explorer modes
                try:
                    output = run_data_exploration_only(
                        query=query,
                        db_id=args.db_id,
                        session_id=session_id,
                        exploration_mode=exploration_mode,
                        llm=llm,
                    )
                    
                    if output["success"]:
                        previous_state = output["state"]  # Save state for next query
                        
                        # If data was generated (text2sql or full_pipeline), ask what to do next
                        if (exploration_mode in ["text2sql", "full_pipeline"] and 
                            previous_state.get("df_raw_redis_key")):
                            
                            next_choice = prompt_next_step_after_data()
                            
                            if next_choice == "1":  # Proceed to causal analysis
                                print("✓ Switching to Causal Analysis mode...")
                                analysis_mode = "causal_analysis"
                                continue
                            elif next_choice == "2":  # Continue exploration
                                print("✓ Continuing data exploration...")
                                continue  # Back to select_data_exploration_mode
                            elif next_choice == "3":  # Exit
                                print("👋 Goodbye!")
                                break
                    else:
                        print(f"❌ Error: {output['state'].get('error')}")
                
                except Exception as e:
                    logger.exception("Data exploration failed")
                    print(f"❌ Error: {e}")
            
            else:
                # Causal Analysis mode: check for previous data first
                if (previous_state and 
                    previous_state.get("data_exploration_status") == "completed" and
                    (previous_state.get("df_raw_redis_key") or previous_state.get("df_clean_redis_key"))):
                    reuse_choice = prompt_data_reuse(previous_state)
                    
                    if reuse_choice == "4":  # Exit
                        print("👋 Goodbye!")
                        break
                    elif reuse_choice == "1":  # Use data for causal analysis
                        print("✓ Proceeding to causal analysis with existing data...")
                        # Continue to query input below
                    elif reuse_choice == "2":  # Explore new data
                        previous_state = None
                        analysis_mode = select_analysis_mode()
                        continue
                    elif reuse_choice == "3":  # Continue current exploration
                        print("✓ Switching to data exploration...")
                        analysis_mode = "data_exploration"
                        continue
                
                # For causal analysis, ask for the causal query
                query = input("\n🧑 Enter your causal analysis query (or 'done' to change mode, 'quit' to exit): ").strip()
                
                # Handle 'done' - return to mode selection
                if query.lower() == "done":
                    print("✓ Returning to mode selection...")
                    analysis_mode = select_analysis_mode()
                    continue
                
                # Handle empty input - stay in current mode
                if not query:
                    continue
                
                # Handle 'quit' - exit program
                if query.lower() in ["exit", "quit", "q"]:
                    if confirm_quit():
                        print("👋 Goodbye!")
                        break
                    else:
                        continue
                
                # Execute full causal analysis pipeline
                try:
                    print_previous_state_summary(previous_state)
                    
                    output = run_full_pipeline(
                        query,
                        db_id=args.db_id,
                        session_id=session_id,
                        previous_state=previous_state,
                        planner_config=planner_cfg,
                        executor_config=executor_cfg,
                        orchestration_config=orchestration_cfg,
                        use_synthetic_df=False,
                    )
                    
                    if output["success"]:
                        logger.info("Success")
                        previous_state = output["state"]  # Save state for next query
                        
                        if args.print_report and output["state"].get("final_report"):
                            fr = output["state"].get("final_report", {})
                            print("\n=== Final Report (Markdown) ===")
                            print(fr.get("markdown", ""))
                    else:
                        logger.error("Failed: %s", output["state"].get("error"))
                        print(f"❌ Error: {output['state'].get('error')}")
                
                except Exception as e:
                    logger.exception("Execution failed")
                    print(f"❌ Error: {e}")
                    # Don't break the loop, allow user to try again
        
        # Cleanup after interactive loop exits
        try:
            from monitoring.metrics.collector import get_metrics_collector
            from core.memory import session_memory
            
            metrics_collector = get_metrics_collector()
            if metrics_collector:
                metrics_collector.stop_monitoring()
                logger.info("Metrics collection stopped")
            
            # Clear session memory for this session
            session_memory.clear_session(session_id)
            logger.info(f"Session memory cleared for {session_id}")
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")