"""
User Study Entry Point

This script is the entry point for user study experiments, supporting two conditions:
1. ORCA: Full causal analysis system with HITL
2. Baseline: Alternative baseline system

Usage example:
    python user_study_entry.py --participant_id P001 --condition orca
    python user_study_entry.py --participant_id P002 --condition baseline
"""

import logging
import argparse
import sys
from datetime import datetime
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)
logger = logging.getLogger(__name__)


def run_orca_condition(
    participant_id: str,
    db_id: str = "reef_db",
    task_id: Optional[str] = None,
    experiment_mode: bool = True,
    run_context: Optional[Any] = None
):
    """
    Run ORCA condition: full interactive causal analysis system.
    
    Args:
        participant_id: Unique participant identifier
        db_id: Database identifier to use
        task_id: Task identifier (for experiment tracking)
        experiment_mode: Whether experiment tracking is enabled
        run_context: Optional RunContext instance (for experiment tracking)
    """
    logger.info(f"ORCA condition called: experiment_mode={experiment_mode}, run_context={run_context is not None}")
    
    from main import (
        select_analysis_mode,
        select_data_exploration_mode,
        run_data_exploration_only,
        run_full_pipeline,
        confirm_quit,
        prompt_data_reuse,
        prompt_next_step_after_data,
        print_previous_state_summary
    )
    from utils.llm import get_llm
    from monitoring.metrics.collector import MetricsCollector, set_metrics_collector
    
    print("\n" + "="*60)
    print("🤖 ORCA: Causal Analysis System (User Study)")
    print("="*60)
    print(f"   Participant ID: {participant_id}")
    print(f"   Condition: ORCA")
    print(f"   Database: {db_id}")
    print("="*60)
    print("\n💡 Navigation Tips:")
    print("   • Type 'done' to return to change modes")
    print("   • Type 'quit' to exit the entire session")
    print("   • After data exploration, you can reuse data for subsequent queries")
    print()
    
    # Initialize session with participant ID
    session_id = f"{participant_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    previous_state: Optional[Dict[str, Any]] = None
    
    planner_cfg: Dict[str, Any] = {}
    executor_cfg: Dict[str, Any] = {}
    orchestration_cfg: Dict[str, Any] = {"interactive": True}
    
    # Get event logger if in experiment mode
    event_logger = None
    if experiment_mode and run_context:
        event_logger = run_context.get_event_logger()
        logger.info(f"Event logger retrieved: {event_logger is not None}")
    
    # Initialize LLM
    llm = get_llm(model="gpt-4o-mini", temperature=0.3, provider="openai")
    
    # Initialize metrics collector if in experiment mode
    if experiment_mode and run_context:
        event_logger_instance = run_context.get_event_logger()
        artifact_manager_instance = run_context.get_artifact_manager()
        logger.info(f"Initializing metrics collector with artifact_manager: {artifact_manager_instance is not None}")
        metrics_collector = MetricsCollector(
            session_id=session_id,
            event_logger=event_logger_instance,
            artifact_manager=artifact_manager_instance
        )
        set_metrics_collector(metrics_collector)
        run_context.set_metrics_collector(metrics_collector)
        metrics_collector.start_monitoring(interval=1.0)
        logger.info("✅ Metrics collection started for experiment mode")
    else:
        logger.warning(f"❌ Metrics collector NOT initialized: experiment_mode={experiment_mode}, run_context={run_context is not None}")
    
    # Step 1: Select analysis mode at session start
    analysis_mode = select_analysis_mode()
    
    # Main query loop (same as main.py interactive mode)
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
                    
                    # Try to get table list from table_relations
                    relations_key = f"{db_id}:table_relations"
                    relations_raw = redis_client.get(relations_key)
                    
                    available_tables = []
                    if relations_raw:
                        relations_info = json.loads(relations_raw)
                        source_schema = relations_info.get("source_schema", {})
                        available_tables = list(source_schema.keys())
                    
                    if available_tables:
                        print(f"\n📋 Available tables in {db_id}:")
                        for i, table in enumerate(sorted(available_tables), 1):
                            print(f"   {i}. {table}")
                        print()
                    else:
                        print(f"\n⚠️  No tables found in database '{db_id}'")
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
                            db_id=db_id,
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
                    db_id=db_id,
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
                    print("✓ Previous data cleared. You can now enter a new causal analysis query.")
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
                    db_id=db_id,
                    session_id=session_id,
                    previous_state=previous_state,
                    planner_config=planner_cfg,
                    executor_config=executor_cfg,
                    orchestration_config=orchestration_cfg,
                    use_synthetic_df=False,
                    event_logger=event_logger,
                )
                
                if output["success"]:
                    logger.info("Success")
                    previous_state = output["state"]  # Save state for next query
                    
                    if output["state"].get("final_report"):
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
    
    print(f"\n✅ Session completed for participant {participant_id}")
    logger.info(f"ORCA condition completed for participant {participant_id}")
    # Explicitly exit to ensure clean termination
    sys.exit(0)


def run_baseline_condition(participant_id: str, db_id: str = "reef_db"):
    """
    Run baseline condition: alternative baseline system.
    
    Args:
        participant_id: Unique participant identifier
        db_id: Database identifier to use
    """
    from baseline.baseline_agent import run_baseline_interactive
    
    print("\n" + "="*60)
    print("🤖 Baseline System (User Study)")
    print("="*60)
    print(f"   Participant ID: {participant_id}")
    print(f"   Condition: Baseline")
    print(f"   Database: {db_id}")
    print("="*60)
    print()
    
    # Initialize session with participant ID
    session_id = f"{participant_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Run baseline interactive mode
    try:
        run_baseline_interactive(
            participant_id=participant_id,
            session_id=session_id,
            db_id=db_id
        )
        print(f"\n✅ Session completed for participant {participant_id}")
        logger.info(f"Baseline condition completed for participant {participant_id}")
    except Exception as e:
        logger.exception(f"Baseline condition failed for participant {participant_id}")
        print(f"❌ Error: {e}")
        sys.exit(1)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="User Study Entry Point")
    parser.add_argument(
        "--participant-id",
        required=True,
        help="Participant ID (e.g., P001, P002)"
    )
    parser.add_argument(
        "--condition",
        required=True,
        choices=["orca", "baseline"],
        help="Study condition: 'orca' or 'baseline'"
    )
    parser.add_argument(
        "--db-id",
        default="reef_db",
        help="Database ID to use (default: reef_db)"
    )
    parser.add_argument(
        "--task-id",
        required=True,
        help="Task ID (e.g., marketing, operations)"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Log level"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    logger.info(f"Starting user study session: Participant={args.participant_id}, Condition={args.condition}")
    
    try:
        # Always use experiment tracking mode
        from monitoring.experiment import RunContext
        
        with RunContext(
            participant_id=args.participant_id,
            condition=args.condition,
            task_id=args.task_id
        ) as run_ctx:
            logger.info("Experiment tracking enabled")
            
            if args.condition == "orca":
                run_orca_condition(
                    args.participant_id,
                    args.db_id,
                    task_id=args.task_id,
                    experiment_mode=True,
                    run_context=run_ctx
                )
            elif args.condition == "baseline":
                run_baseline_condition(args.participant_id, args.db_id)
                
    except KeyboardInterrupt:
        print("\n\n⚠️  Session interrupted by user (Ctrl+C)")
        logger.info(f"Session interrupted for participant {args.participant_id}")
        sys.exit(0)
    except Exception as e:
        logger.exception(f"User study session failed for participant {args.participant_id}")
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)
    
    # Normal completion
    sys.exit(0)

