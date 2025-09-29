# utils/system_agents.py
"""
System-level agents for database connection and metadata creation.

These are not traditional agents but system initialization components
that handle database connection and metadata generation. They are
called during agent initialization and provide essential system setup.
"""

from typing import Any, Dict, Optional, List
import logging
from datetime import datetime

from .database import Database
from .data_prep.metadata import generate_metadata, extract_schema, update_table_relations
from .data_prep.related_tables import update_table_relations as update_relations

logger = logging.getLogger(__name__)

class DatabaseAgent:
    """
    Database connection agent - handles database connection and basic operations.
    
    This is a system component that manages database connections and provides
    basic database information for other agents to use.
    """
    
    def __init__(self, db_id: str, db_type: str = "postgresql", config: Optional[Dict] = None):
        """
        Initialize database agent.
        
        Args:
            db_id: Database identifier
            db_type: Type of database (postgresql, sqlite)
            config: Database configuration
        """
        self.db_id = db_id
        self.db_type = db_type
        self.config = config
        self.database = Database(db_type=db_type, config=config)
        self.connection = None
        self.is_connected = False
        
        logger.info(f"DatabaseAgent initialized for database: {db_id}")
    
    def connect(self) -> bool:
        """
        Establish database connection.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            self.connection = self.database.get_connection(self.db_id)
            self.is_connected = True
            logger.info(f"Successfully connected to database: {self.db_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to database {self.db_id}: {str(e)}")
            self.is_connected = False
            return False
    
    def disconnect(self) -> None:
        """Close database connection."""
        if self.connection:
            try:
                self.connection.close()
                self.is_connected = False
                logger.info(f"Disconnected from database: {self.db_id}")
            except Exception as e:
                logger.error(f"Error disconnecting from database: {str(e)}")
    
    def get_connection(self):
        """Get database connection object."""
        if not self.is_connected:
            self.connect()
        return self.connection
    
    def test_connection(self) -> Dict[str, Any]:
        """
        Test database connection and return status.
        
        Returns:
            Dictionary with connection status and info
        """
        try:
            if not self.is_connected:
                self.connect()
            
            # Test with a simple query
            rows, columns = self.database.run_query("SELECT 1 as test", self.db_id)
            
            return {
                "status": "connected",
                "db_id": self.db_id,
                "db_type": self.db_type,
                "test_query_result": rows[0][0] if rows else None,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "status": "failed",
                "db_id": self.db_id,
                "db_type": self.db_type,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def get_database_info(self) -> Dict[str, Any]:
        """
        Get comprehensive database information.
        
        Returns:
            Dictionary with database metadata
        """
        try:
            if not self.is_connected:
                self.connect()
            
            # Get database version
            version_query = "SELECT version()" if self.db_type == "postgresql" else "SELECT sqlite_version()"
            version_rows, _ = self.database.run_query(version_query, self.db_id)
            version = version_rows[0][0] if version_rows else "Unknown"
            
            # Get table count
            table_count_query = """
            SELECT COUNT(*) 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            """ if self.db_type == "postgresql" else """
            SELECT COUNT(*) 
            FROM sqlite_master 
            WHERE type='table'
            """
            table_rows, _ = self.database.run_query(table_count_query, self.db_id)
            table_count = table_rows[0][0] if table_rows else 0
            
            return {
                "db_id": self.db_id,
                "db_type": self.db_type,
                "version": version,
                "table_count": table_count,
                "is_connected": self.is_connected,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to get database info: {str(e)}")
            return {
                "db_id": self.db_id,
                "db_type": self.db_type,
                "error": str(e),
                "is_connected": False,
                "timestamp": datetime.now().isoformat()
            }

class MetadataAgent:
    """
    Metadata creation agent - handles database schema and metadata generation.
    
    This is a system component that generates comprehensive metadata about
    the database schema, table relationships, and data characteristics.
    """
    
    def __init__(self, database_agent: DatabaseAgent):
        """
        Initialize metadata agent.
        
        Args:
            database_agent: Connected database agent instance
        """
        self.database_agent = database_agent
        self.schema_info = None
        self.table_metadata = None
        self.table_relations = None
        self.is_initialized = False
        
        logger.info("MetadataAgent initialized")
    
    def initialize(self) -> bool:
        """
        Initialize metadata by generating schema and table information.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            logger.info("Starting metadata initialization...")
            
            # Step 1: Extract schema information
            self.schema_info = self._extract_schema()
            logger.info(f"Schema extraction completed: {len(self.schema_info.get('tables', {}))} tables")
            
            # Step 2: Generate table metadata
            self.table_metadata = self._generate_table_metadata()
            logger.info(f"Table metadata generation completed: {len(self.table_metadata)} tables")
            
            # Step 3: Update table relations
            self.table_relations = self._update_table_relations()
            logger.info(f"Table relations updated: {len(self.table_relations)} relations")
            
            self.is_initialized = True
            logger.info("Metadata initialization completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Metadata initialization failed: {str(e)}")
            self.is_initialized = False
            return False
    
    def _extract_schema(self) -> Dict[str, Any]:
        """Extract database schema information."""
        try:
            return extract_schema(self.database_agent.database, self.database_agent.db_id)
        except Exception as e:
            logger.error(f"Schema extraction failed: {str(e)}")
            return {"tables": {}, "error": str(e)}
    
    def _generate_table_metadata(self) -> Dict[str, Any]:
        """Generate comprehensive table metadata."""
        try:
            return generate_metadata(self.database_agent.database, self.database_agent.db_id)
        except Exception as e:
            logger.error(f"Table metadata generation failed: {str(e)}")
            return {"error": str(e)}
    
    def _update_table_relations(self) -> Dict[str, Any]:
        """Update table relationship information."""
        try:
            return update_relations(self.database_agent.database, self.database_agent.db_id)
        except Exception as e:
            logger.error(f"Table relations update failed: {str(e)}")
            return {"error": str(e)}
    
    def get_schema_info(self) -> Dict[str, Any]:
        """Get extracted schema information."""
        if not self.is_initialized:
            self.initialize()
        return self.schema_info or {}
    
    def get_table_metadata(self) -> Dict[str, Any]:
        """Get generated table metadata."""
        if not self.is_initialized:
            self.initialize()
        return self.table_metadata or {}
    
    def get_table_relations(self) -> Dict[str, Any]:
        """Get table relationship information."""
        if not self.is_initialized:
            self.initialize()
        return self.table_relations or {}
    
    def get_metadata_summary(self) -> Dict[str, Any]:
        """Get comprehensive metadata summary."""
        if not self.is_initialized:
            self.initialize()
        
        return {
            "is_initialized": self.is_initialized,
            "schema_info": self.schema_info,
            "table_metadata": self.table_metadata,
            "table_relations": self.table_relations,
            "timestamp": datetime.now().isoformat()
        }
    
    def refresh_metadata(self) -> bool:
        """Refresh all metadata by re-initializing."""
        logger.info("Refreshing metadata...")
        self.is_initialized = False
        return self.initialize()

def create_system_agents(db_id: str, db_type: str = "postgresql", 
                        config: Optional[Dict] = None) -> tuple[DatabaseAgent, MetadataAgent]:
    """
    Create and initialize system agents.
    
    Args:
        db_id: Database identifier
        db_type: Database type
        config: Database configuration
        
    Returns:
        Tuple of (DatabaseAgent, MetadataAgent) instances
    """
    # Create database agent
    db_agent = DatabaseAgent(db_id, db_type, config)
    
    # Connect to database
    if not db_agent.connect():
        raise Exception(f"Failed to connect to database: {db_id}")
    
    # Create metadata agent
    metadata_agent = MetadataAgent(db_agent)
    
    # Initialize metadata
    if not metadata_agent.initialize():
        raise Exception("Failed to initialize metadata")
    
    logger.info("System agents created and initialized successfully")
    return db_agent, metadata_agent
