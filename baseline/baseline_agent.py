"""
Baseline Agent for User Study

This is the baseline condition for the user study, providing an alternative
approach to causal analysis without the ORCA system's features.

The baseline can be customized based on your study design needs.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


def run_baseline_interactive(
    participant_id: str,
    session_id: str,
    db_id: str = "reef_db"
) -> None:
    """
    Run baseline interactive mode for user study.
    
    This is a placeholder implementation. Customize based on your baseline needs.
    
    Args:
        participant_id: Unique participant identifier
        session_id: Session identifier
        db_id: Database identifier
    """
    print("\n💡 Baseline System Instructions:")
    print("   • This is the baseline condition")
    print("   • Type 'quit' to exit the session")
    print()
    
    # Track session data
    session_data = {
        "participant_id": participant_id,
        "session_id": session_id,
        "db_id": db_id,
        "start_time": datetime.now().isoformat(),
        "queries": []
    }
    
    # Main interaction loop
    while True:
        print("\n" + "-"*60)
        
        # Get user query
        query = input("\n🧑 Enter your query (or 'quit' to exit): ").strip()
        
        # Handle quit
        if not query or query.lower() in ["exit", "quit", "q"]:
            if confirm_quit_baseline():
                break
            else:
                continue
        
        # Process query
        try:
            print(f"\n🔄 Processing query: {query}")
            
            # TODO: Implement your baseline logic here
            # This could be:
            # 1. A simpler version of causal analysis
            # 2. A different causal analysis tool/method
            # 3. Manual step-by-step prompts
            # 4. Other baseline approach based on your study design
            
            result = process_baseline_query(query, db_id, session_id)
            
            # Log query
            session_data["queries"].append({
                "query": query,
                "timestamp": datetime.now().isoformat(),
                "result": result
            })
            
            # Display result
            if result.get("success"):
                print("\n✅ Query processed successfully")
                if result.get("output"):
                    print(f"\n{result['output']}")
            else:
                print(f"\n❌ Error: {result.get('error', 'Unknown error')}")
        
        except Exception as e:
            logger.exception(f"Query processing failed: {e}")
            print(f"❌ Error: {e}")
    
    # Save session data
    session_data["end_time"] = datetime.now().isoformat()
    save_session_data(session_data)
    
    print("\n✅ Baseline session completed")


def confirm_quit_baseline() -> bool:
    """
    Ask user to confirm they want to quit the baseline session.
    
    Returns:
        True if user confirms quit, False otherwise
    """
    print("\n" + "="*60)
    print("⚠️  WARNING: Exiting will end your session!")
    print("="*60)
    confirm = input("❓ Are you sure you want to quit? (yes/no): ").strip().lower()
    if confirm in ["yes", "y"]:
        return True
    else:
        print("✓ Continuing session...")
        return False


def process_baseline_query(
    query: str,
    db_id: str,
    session_id: str
) -> Dict[str, Any]:
    """
    Process a query in the baseline system.
    
    TODO: Implement your baseline logic here.
    
    Args:
        query: User query
        db_id: Database identifier
        session_id: Session identifier
    
    Returns:
        Dictionary with processing results
    """
    # Placeholder implementation
    # Replace with your actual baseline logic
    
    try:
        # Example placeholder response
        output = f"""
Baseline Analysis Results for: "{query}"

[TODO: Implement baseline analysis logic]

This is where you would:
1. Process the user's query
2. Perform causal analysis (using your baseline method)
3. Return results in a comparable format to ORCA

Note: This is a placeholder. Customize based on your study design.
"""
        
        return {
            "success": True,
            "output": output,
            "query": query,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.exception(f"Baseline query processing failed: {e}")
        return {
            "success": False,
            "error": str(e)
        }


def save_session_data(session_data: Dict[str, Any]) -> None:
    """
    Save session data for later analysis.
    
    Args:
        session_data: Session data to save
    """
    try:
        import json
        from pathlib import Path
        
        # Create logs directory if it doesn't exist
        log_dir = Path("logs/user_study")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Save session data
        participant_id = session_data.get("participant_id", "unknown")
        session_id = session_data.get("session_id", "unknown")
        filename = f"baseline_{participant_id}_{session_id}.json"
        filepath = log_dir / filename
        
        with open(filepath, "w") as f:
            json.dump(session_data, f, indent=2)
        
        logger.info(f"Session data saved to {filepath}")
        print(f"\n📝 Session log saved: {filepath}")
    
    except Exception as e:
        logger.exception(f"Failed to save session data: {e}")
        print(f"⚠️  Warning: Could not save session data: {e}")

