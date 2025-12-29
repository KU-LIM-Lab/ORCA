# utils/system_init.py
"""
Simple system initialization for database connection and metadata generation.

This module provides lightweight functions for:
1. Database connection
2. Metadata generation with Redis caching
3. System initialization before main agent creation
"""

from typing import Any, Dict, Optional
import logging
import json
from datetime import datetime

from .database import Database
from .data_prep.metadata import generate_metadata, extract_schema
from .data_prep.related_tables import update_table_relations
from .redis_client import redis_client



logger = logging.getLogger(__name__)

class SystemInitializer:
    """Simple system initializer for database and metadata"""
    
    def __init__(self, db_id: str, db_type: str = "postgresql", config: Optional[Dict[str, Any]] = None):
        self.db_id = db_id
        self.db_type = db_type
        self.config = config or {}
        self.database = Database(db_type=db_type, config=config)
        self.connection = None
        self.is_connected = False
        
    def connect_database(self) -> bool:
        """Connect to database"""
        try:
            self.connection = self.database.get_connection(self.db_id)
            self.is_connected = True
            logger.info(f"Database connected: {self.db_id}")
            return True
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            self.is_connected = False
            return False
    
    def get_connection(self):
        """Get database connection"""
        if not self.is_connected:
            self.connect_database()
        return self.connection
    
    def ensure_metadata(self) -> Dict[str, Any]:
        """Ensure metadata exists in Redis, create if missing"""
        try:
            # Check if metadata exists in Redis
            redis_key = f"{self.db_id}:metadata"
            metadata_raw = redis_client.get(redis_key)
            
            if metadata_raw:
                logger.info(f"Metadata found in Redis for {self.db_id}")
                return json.loads(metadata_raw)
            
            # Generate metadata if not exists
            logger.info(f"Generating metadata for {self.db_id}")
            return self._generate_metadata()
            
        except Exception as e:
            logger.error(f"Metadata generation failed: {e}")
            return {"error": str(e)}
    
    def _generate_metadata(self) -> Dict[str, Any]:
        """Generate metadata and store in Redis"""
        try:
            # Extract schema
            schema_info = extract_schema(self.db_id)
            logger.info(f"Schema extracted: {len(schema_info.get('tables', {}))} tables")
            
            # Generate table metadata
            table_metadata = {}
            for table_name, table_schema in schema_info.get("tables", {}).items():
                table_metadata[table_name] = generate_metadata(table_name, table_schema)
            
            # Update table relations
            table_relations = update_table_relations(self.db_id)
            
            # Combine all metadata
            metadata = {
                "schema_info": schema_info,
                "table_metadata": table_metadata,
                "table_relations": table_relations,
                "generated_at": datetime.now().isoformat(),
                "db_id": self.db_id
            }
            
            # Store in Redis
            redis_key = f"{self.db_id}:metadata"
            redis_client.set(redis_key, json.dumps(metadata, indent=2, default=str))
            
            # Store individual table metadata for easy access
            for table_name, table_meta in table_metadata.items():
                table_key = f"{self.db_id}:metadata:{table_name}"
                redis_client.set(table_key, json.dumps({
                    "schema": schema_info["tables"][table_name],
                    "metadata": table_meta
                }, indent=2, default=str))
            
            logger.info(f"Metadata generated and stored in Redis for {self.db_id}")
            return metadata
            
        except Exception as e:
            logger.error(f"Metadata generation failed: {e}")
            return {"error": str(e)}
    
    def get_table_metadata(self, table_name: str) -> Optional[Dict[str, Any]]:
        """Get metadata for specific table from Redis"""
        try:
            redis_key = f"{self.db_id}:metadata:{table_name}"
            metadata_raw = redis_client.get(redis_key)
            
            if metadata_raw:
                return json.loads(metadata_raw)
            return None
            
        except Exception as e:
            logger.error(f"Failed to get table metadata for {table_name}: {e}")
            return None
    
    def refresh_metadata(self) -> Dict[str, Any]:
        """Force refresh metadata"""
        logger.info(f"Refreshing metadata for {self.db_id}")
        return self._generate_metadata()

def initialize_system(db_id: str, db_type: str = "postgresql", 
                     config: Optional[Dict[str, Any]] = None) -> SystemInitializer:
    """
    Initialize system with database connection and metadata generation.
    
    This should be called before creating the main agent to ensure
    database connection and metadata are ready.
    
    Args:
        db_id: Database identifier
        db_type: Database type (postgresql, sqlite)
        config: Database configuration
        
    Returns:
        SystemInitializer instance
    """
    logger.info(f"Initializing system for database: {db_id}")
    
    initializer = SystemInitializer(db_id, db_type, config)
    
    # Connect to database
    if not initializer.connect_database():
        raise Exception(f"Failed to connect to database: {db_id}")
    
    # Ensure metadata exists
    metadata = initializer.ensure_metadata()
    if "error" in metadata:
        logger.warning(f"Metadata generation had issues: {metadata['error']}")
    
    logger.info(f"System initialization completed for {db_id}")
    return initializer

def get_table_metadata(db_id: str, table_name: str) -> Optional[Dict[str, Any]]:
    """Get table metadata from Redis"""
    try:
        redis_key = f"{db_id}:metadata:{table_name}"
        metadata_raw = redis_client.get(redis_key)
        
        if metadata_raw:
            return json.loads(metadata_raw)
        return None
        
    except Exception as e:
        logger.error(f"Failed to get table metadata: {e}")
        return None

def check_metadata_exists(db_id: str) -> bool:
    """Check if metadata exists in Redis"""
    try:
        redis_key = f"{db_id}:metadata"
        return redis_client.exists(redis_key)
    except Exception as e:
        logger.error(f"Failed to check metadata existence: {e}")
        return False
