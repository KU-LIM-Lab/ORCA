# main.py
"""
ORCA Main Agent Execution Entry Point

Simple agent execution with debugging points for the new architecture:
- System initialization (database + metadata)
- Planner-Executor orchestration with HITL
- Tool registry pattern for specialist agents
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from datetime import datetime

# Core imports
from core.state import create_initial_state, AgentState
from orchestration.graph import create_orchestration_graph
from monitoring import get_unified_monitor
from utils.system_init import initialize_system

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ORCAMainAgent:
    """
    Main ORCA Agent - with debugging points.
    
    Architecture:
    1. System initialization (database + metadata) - DONE at startup
    2. Planner-Executor orchestration with HITL
    3. Tool registry for specialist agents
    """
    
    def __init__(self, 
                 db_id: str,
                 db_type: str = "postgresql",
                 db_config: Optional[Dict[str, Any]] = None,
                 planner_config: Optional[Dict[str, Any]] = None,
                 executor_config: Optional[Dict[str, Any]] = None):
        """
        Initialize ORCA Main Agent with system pre-initialization.
        
        Args:
            db_id: Database identifier
            db_type: Database type (postgresql, sqlite)
            db_config: Database configuration
            planner_config: Planner agent configuration
            executor_config: Executor agent configuration
        """
        self.db_id = db_id
        self.db_type = db_type
        self.db_config = db_config or {}
        
        logger.info(f"🚀 Initializing ORCA Main Agent for database: {db_id}")
        
        # Initialize monitoring
        self.monitor = get_unified_monitor(f"orca_session_{db_id}")
        self.monitor.start_monitoring()
        
        # DEBUG POINT 1: System initialization (happens at startup)
        logger.info("🔧 DEBUG POINT 1: System Initialization")
        self._initialize_system()
        
        # Initialize orchestration graph
        logger.info("🔧 DEBUG POINT 2: Orchestration Graph Creation")
        self.orchestration_graph = create_orchestration_graph(
            planner_config=planner_config,
            executor_config=executor_config,
            metrics_collector=self.monitor.metrics_collector
        )
        
        logger.info(f"✅ ORCA Main Agent initialized successfully for database: {db_id}")
    
    def _initialize_system(self) -> None:
        """Initialize system components during main agent creation"""
        try:
            logger.info(f"🔧 Initializing system for database: {self.db_id}")
            
            # System initialization (database connection + metadata generation)
            self.system_initializer = initialize_system(
                self.db_id, 
                self.db_type, 
                self.db_config
            )
            
            # DEBUG: Check system status
            if self.system_initializer and self.system_initializer.is_connected:
                logger.info("✅ System initialization completed successfully")
                logger.info(f"   - Database: {self.db_id} ({self.db_type})")
                logger.info(f"   - Metadata: Available in Redis")
            else:
                logger.error("❌ System initialization failed")
                raise Exception("System initialization failed")
                
        except Exception as e:
            logger.error(f"❌ System initialization failed: {e}")
            self.system_initializer = None
            raise
    
    async def execute_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute a user query through the complete ORCA pipeline.
        
        Args:
            query: User's analysis query
            context: Additional context for the analysis
            
        Returns:
            Complete analysis results
        """
        logger.info(f"🎯 Starting query execution: '{query}'")
        
        try:
            with self.monitor.track_execution("orca_main", "query_execution"):
                # DEBUG POINT 3: Initial state creation
                logger.info("🔧 DEBUG POINT 3: Initial State Creation")
                initial_state = create_initial_state(query, self.db_id)
                if context:
                    initial_state.update(context)
                
                # Add system components to state
                initial_state.update({
                    "system_initialized": True,
                    "db_id": self.db_id,
                    "db_type": self.db_type
                })
                
                logger.info(f"   - Initial state keys: {list(initial_state.keys())}")
                
                # DEBUG POINT 4: Orchestration graph execution
                logger.info("🔧 DEBUG POINT 4: Orchestration Graph Execution")
                logger.info("   - Starting Planner-Executor pipeline...")
                
                # Execute orchestration graph
                result_state = self.orchestration_graph.execute(query, initial_state)
                
                # DEBUG POINT 5: Results extraction
                logger.info("🔧 DEBUG POINT 5: Results Extraction")
                results = self._extract_results(result_state)
                
                logger.info("✅ Query execution completed successfully")
                return results
                
        except Exception as e:
            logger.error(f"❌ Query execution failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "query": query,
                "timestamp": datetime.now().isoformat()
            }
    
    def _extract_results(self, state: AgentState) -> Dict[str, Any]:
        """Extract and format results from final state"""
        return {
            "success": True,
            "query": state.get("initial_query", ""),
            "execution_status": state.get("execution_status", ""),
            "results": {
                "data_exploration": {
                    "selected_tables": state.get("selected_tables", []),
                    "sql_query": state.get("sql_query", ""),
                    "data_retrieved": bool(state.get("df_raw")),
                },
                "causal_discovery": {
                    "algorithms_used": state.get("selected_algorithms", []),
                    "causal_graph": state.get("selected_graph", {}),
                },
                "causal_inference": {
                    "treatment_variable": state.get("treatment_variable", ""),
                    "outcome_variable": state.get("outcome_variable", ""),
                    "causal_estimates": state.get("causal_estimates", {}),
                }
            },
            "execution_log": state.get("execution_log", []),
            "hitl_interactions": state.get("hitl_interactions", []),
            "timestamp": datetime.now().isoformat()
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return {
            "system_initialized": self.system_initializer is not None and self.system_initializer.is_connected,
            "database_id": self.db_id,
            "database_type": self.db_type,
            "monitoring_active": self.monitor is not None,
            "timestamp": datetime.now().isoformat()
        }
    
    def shutdown(self) -> None:
        """Shutdown the agent and cleanup resources"""
        logger.info("🛑 Shutting down ORCA Main Agent")
        if self.monitor:
            self.monitor.stop_monitoring()
        logger.info("✅ Shutdown completed")

# Example usage and execution flow
async def main():
    """
    Example of how to run ORCA Main Agent with debugging points.
    
    This demonstrates the complete execution flow from initialization
    to query processing and result generation.
    """
    
    logger.info("=" * 60)
    logger.info("🚀 ORCA Main Agent - Simple Execution Example")
    logger.info("=" * 60)
    
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
    
    try:
        # DEBUG POINT 0: Agent Creation
        logger.info("🔧 DEBUG POINT 0: Agent Creation")
        orca = ORCAMainAgent(
            db_id="reef_db",
            db_type="postgresql",
            db_config=db_config,
            planner_config=planner_config,
            executor_config=executor_config
        )
        
        # Check system status
        status = orca.get_status()
        logger.info(f"📊 System Status: {status}")
        
        if not status["system_initialized"]:
            logger.error("❌ System not properly initialized")
            return
        
        # DEBUG POINT 6: Query Execution
        logger.info("🔧 DEBUG POINT 6: Query Execution")
        sample_queries = [
            "고객 이탈에 영향을 미치는 요인을 분석하고 싶습니다",
            "사용자 행동과 구매 패턴의 인과관계를 알아보고 싶습니다",
            "마케팅 캠페인이 매출에 미치는 영향을 분석해주세요"
        ]
        
        for i, query in enumerate(sample_queries, 1):
            logger.info(f"\n--- Query {i}: {query} ---")
            
            # Execute query
            results = await orca.execute_query(query)
            
            # Display results
            if results["success"]:
                logger.info("✅ Query executed successfully!")
                logger.info(f"   - Selected tables: {results['results']['data_exploration']['selected_tables']}")
                logger.info(f"   - SQL query: {results['results']['data_exploration']['sql_query']}")
                logger.info(f"   - Execution log entries: {len(results['execution_log'])}")
            else:
                logger.error(f"❌ Query execution failed: {results['error']}")
        
        # Final status
        logger.info("\n📊 Final System Status:")
        final_status = orca.get_status()
        logger.info(f"   - System initialized: {final_status['system_initialized']}")
        logger.info(f"   - Monitoring active: {final_status['monitoring_active']}")
        
    except Exception as e:
        logger.error(f"❌ Main execution failed: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
    
    finally:
        # Cleanup
        logger.info("\n🛑 Cleaning up...")
        if 'orca' in locals():
            orca.shutdown()
        logger.info("✅ Cleanup completed")

if __name__ == "__main__":
    # Run the main execution
    asyncio.run(main())