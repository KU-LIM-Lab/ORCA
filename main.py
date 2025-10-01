# main.py
"""
ORCA Main Agent Execution Entry Point

This is the main entry point for running the ORCA system.
It demonstrates how to initialize and execute the complete pipeline.
"""

import asyncio
import logging
from typing import Dict, Any, Optional

# Core imports
from core.state import create_initial_state, AgentState
from orchestration.graph import OrchestrationGraph, create_orchestration_graph
from monitoring import get_unified_monitor, set_metrics_collector
from utils.tools import tool_registry, DatabaseTool

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ORCAMainAgent:
    """
    Main ORCA Agent that orchestrates the entire analysis pipeline.
    
    This agent coordinates all phases of the causal analysis workflow:
    1. System initialization (database, metadata)
    2. Data exploration
    3. Causal discovery
    4. Causal inference
    5. Report generation
    """
    
    def __init__(self, 
                 db_id: str,
                 db_type: str = "postgresql",
                 db_config: Optional[Dict[str, Any]] = None,
                 planner_config: Optional[Dict[str, Any]] = None,
                 executor_config: Optional[Dict[str, Any]] = None):
        """
        Initialize ORCA Main Agent.
        
        Args:
            db_id: Database identifier
            db_type: Database type (postgresql, sqlite)
            db_config: Database configuration
            planner_config: Planner agent configuration
            executor_config: Executor agent configuration
        """
        self.db_id = db_id
        self.db_type = db_type
        self.db_config = db_config
        
        # Initialize monitoring
        self.monitor = get_unified_monitor(f"orca_session_{db_id}")
        self.monitor.start_monitoring()
        
        # Initialize system before creating orchestration graph
        self.system_initializer = None
        self._initialize_system()
        
        # Initialize orchestration graph
        self.orchestration_graph = create_orchestration_graph(
            planner_config=planner_config,
            executor_config=executor_config,
            metrics_collector=self.monitor.metrics_collector
        )
        
        logger.info(f"ORCA Main Agent initialized for database: {db_id}")
    
    def _initialize_system(self) -> None:
        """Initialize system components during main agent creation"""
        try:
            from utils.system_init import initialize_system
            
            logger.info(f"Initializing system for database: {self.db_id}")
            self.system_initializer = initialize_system(
                self.db_id, 
                self.db_type, 
                self.db_config
            )
            logger.info("System initialization completed")
            
        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            self.system_initializer = None
    
    async def initialize_system(self) -> bool:
        """
        Check if system is already initialized.
        
        Returns:
            True if system is ready, False otherwise
        """
        if self.system_initializer and self.system_initializer.is_connected:
            logger.info("System already initialized")
            return True
        
        logger.error("System not initialized")
        return False
    
    async def execute_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute a user query through the complete ORCA pipeline.
        
        Args:
            query: User's analysis query
            context: Additional context for the analysis
            
        Returns:
            Complete analysis results
        """
        try:
            logger.info(f"Executing query: {query}")
            
            with self.monitor.track_execution("orca_main", "query_execution"):
                # Step 1: Create initial state
                initial_state = create_initial_state(query, self.db_id)
                if context:
                    initial_state.update(context)
                
                # Step 2: Add system components to state
                initial_state.update({
                    "system_initializer": self.system_initializer,
                    "system_initialized": True,
                    "db_id": self.db_id,
                    "db_type": self.db_type
                })
                
                # Step 3: Execute orchestration graph
                result_state = self.orchestration_graph.execute(query, initial_state)
                
                # Step 4: Extract results
                results = self._extract_results(result_state)
                
                logger.info("Query execution completed successfully")
                return results
                
        except Exception as e:
            logger.error(f"Query execution failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "query": query
            }
    
    def _extract_results(self, state: AgentState) -> Dict[str, Any]:
        """Extract and format results from final state."""
        return {
            "success": True,
            "query": state.get("initial_query", ""),
            "execution_status": state.get("execution_status", ""),
            "results": {
                "data_exploration": state.get("data_exploration_results", {}),
                "causal_discovery": state.get("causal_discovery_results", {}),
                "causal_inference": state.get("causal_inference_results", {}),
                "final_report": state.get("final_report", {})
            },
            "metadata": {
                "execution_time": state.get("total_execution_time", 0),
                "phases_completed": state.get("phases_completed", []),
                "hitl_interactions": state.get("hitl_interactions", [])
            },
            "monitoring": self.monitor.get_unified_summary()
        }
    
    async def handle_hitl_response(self, state: AgentState, 
                                 user_decision: str,
                                 user_edits: Optional[Dict[str, Any]] = None,
                                 user_feedback: Optional[str] = None) -> Dict[str, Any]:
        """
        Handle Human-in-the-Loop response and continue execution.
        
        Args:
            state: Current state
            user_decision: User's decision (approve/edit/feedback/abort)
            user_edits: User-provided edits
            user_feedback: User feedback
            
        Returns:
            Updated results
        """
        try:
            logger.info(f"Handling HITL response: {user_decision}")
            
            # Continue execution with user input
            updated_state = self.orchestration_graph.handle_hitl_response(
                state, user_decision, user_edits, user_feedback
            )
            
            # Extract updated results
            results = self._extract_results(updated_state)
            
            logger.info("HITL response handled successfully")
            return results
            
        except Exception as e:
            logger.error(f"HITL response handling failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "user_decision": user_decision
            }
    
    def get_status(self) -> Dict[str, Any]:
        """Get current system status."""
        return {
            "db_id": self.db_id,
            "db_type": self.db_type,
            "system_initialized": self.db_agent is not None and self.metadata_agent is not None,
            "orchestration_status": self.orchestration_graph.get_status(),
            "monitoring": self.monitor.get_unified_summary(),
            "tools_registered": tool_registry.get_usage_stats()
        }
    
    def shutdown(self):
        """Shutdown the system gracefully."""
        try:
            logger.info("Shutting down ORCA system...")
            
            # Disconnect from database
            if self.db_agent:
                self.db_agent.disconnect()
            
            # Stop monitoring
            self.monitor.stop_monitoring()
            
            logger.info("ORCA system shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {str(e)}")

# Example usage and execution flow
async def main():
    """
    Example of how to run ORCA Main Agent.
    
    This demonstrates the complete execution flow from initialization
    to query processing and result generation.
    """
    
    # Configuration
    db_config = {
        "host": "localhost",
        "port": 5432,
        "user": "postgres",
        "password": "password"
    }
    
    planner_config = {
        "model": "gpt-4o-mini",
        "temperature": 0.3
    }
    
    executor_config = {
        "max_retries": 3,
        "timeout": 300
    }
    
    # Initialize ORCA Main Agent
    orca = ORCAMainAgent(
        db_id="daa",
        db_type="postgresql",
        db_config=db_config,
        planner_config=planner_config,
        executor_config=executor_config
    )
    
    try:
        # System is already initialized during ORCAMainAgent creation
        logger.info("=== Step 1: System Status ===")
        if not orca.system_initializer or not orca.system_initializer.is_connected:
            logger.error("System initialization failed")
            return
        
        # Step 2: Execute sample query
        logger.info("=== Step 2: Query Execution ===")
        sample_query = "고객 이탈에 영향을 미치는 요인을 분석하고 싶습니다"
        
        results = await orca.execute_query(sample_query)
        
        if results["success"]:
            logger.info("Query executed successfully!")
            logger.info(f"Results: {results['results']}")
        else:
            logger.error(f"Query execution failed: {results['error']}")
        
        # Step 3: Show system status
        logger.info("=== Step 3: System Status ===")
        status = orca.get_status()
        logger.info(f"System Status: {status}")
        
    except Exception as e:
        logger.error(f"Main execution failed: {str(e)}")
    
    finally:
        # Step 4: Shutdown
        logger.info("=== Step 4: System Shutdown ===")
        orca.shutdown()

if __name__ == "__main__":
    # Run the main execution
    asyncio.run(main())
