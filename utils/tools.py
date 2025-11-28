# utils/tools.py
"""
Unified tool system for ORCA agents.

This module provides a standardized interface for tools that agents can use,
integrating with existing utilities in the utils package. Tools are designed
to be lightweight wrappers around core functionality with consistent interfaces.

Key Features:
- BaseTool: Abstract base class for all tools
- ToolRegistry: Central registry for tool management
- Integration with existing utils (database, llm, etc.)
- Execution tracking and monitoring
- Consistent error handling and logging
"""

from typing import Any, Dict, List, Optional, Callable, Union
from abc import ABC, abstractmethod
import json
import pandas as pd
from datetime import datetime
import asyncio
from functools import wraps
import logging

# Import existing utilities - these are the core implementations
from .database import Database
from .llm import call_llm, call_llm_async, get_llm

logger = logging.getLogger(__name__)

class BaseTool(ABC):
    """
    Abstract base class for all tools.
    
    Tools provide a standardized interface for agents to access various
    functionalities. Each tool should be stateless and thread-safe.
    """
    
    def __init__(self, name: str, description: str = ""):
        """
        Initialize the tool.
        
        Args:
            name: Unique name for the tool
            description: Human-readable description of what the tool does
        """
        self.name = name
        self.description = description
        self.usage_count = 0
        self.last_used = None
        self._lock = asyncio.Lock()  # For thread safety
    
    @abstractmethod
    def execute(self, *args, **kwargs) -> Any:
        """
        Execute the tool with given arguments.
        
        This method should be implemented by subclasses to provide
        the actual tool functionality.
        """
        pass
    
    def __call__(self, *args, **kwargs) -> Any:
        """
        Make the tool callable like a function.
        
        This method handles usage tracking and delegates to execute().
        """
        self.usage_count += 1
        self.last_used = datetime.now()
        return self.execute(*args, **kwargs)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize tool metadata to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "usage_count": self.usage_count,
            "last_used": self.last_used.isoformat() if self.last_used else None
        }
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics for this tool."""
        return {
            "name": self.name,
            "usage_count": self.usage_count,
            "last_used": self.last_used.isoformat() if self.last_used else None,
            "is_active": self.last_used is not None
        }

class DatabaseTool(BaseTool):
    """
    Database tool providing SQL query execution capabilities.
    
    This tool wraps the existing Database class from utils.database
    and provides a standardized interface for database operations.
    """
    
    def __init__(self, db_id: str, db_type: str = "postgresql", config: Optional[Dict] = None):
        """
        Initialize database tool.
        
        Args:
            db_id: Database identifier
            db_type: Type of database (postgresql, sqlite)
            config: Database configuration dictionary
        """
        super().__init__("database", "Database connection and query execution")
        self.db_id = db_id
        self.db = Database(db_type=db_type, config=config)
        logger.info(f"DatabaseTool initialized for database: {db_id}")
    
    def execute(self, query: str, params: Optional[Dict] = None) -> pd.DataFrame:
        """
        Execute SQL query and return results as DataFrame.
        
        Args:
            query: SQL query string
            params: Optional query parameters
            
        Returns:
            pandas.DataFrame: Query results
            
        Raises:
            Exception: If query execution fails
        """
        try:
            logger.debug(f"Executing query on database {self.db_id}")
            rows, columns = self.db.run_query(query, self.db_id)
            df = pd.DataFrame(rows, columns=columns)
            logger.info(f"Query executed successfully, returned {len(df)} rows")
            return df
        except Exception as e:
            logger.error(f"Database query failed: {str(e)}")
            raise Exception(f"Database query failed: {str(e)}")
    
    def get_connection(self):
        """Get raw database connection object."""
        return self.db.get_connection(self.db_id)
    
    def list_tables(self) -> List[str]:
        """
        List all tables in the database.
        
        Returns:
            List of table names
        """
        query = """
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public'
        """
        try:
            df = self.execute(query)
            tables = df['table_name'].tolist()
            logger.info(f"Found {len(tables)} tables in database {self.db_id}")
            return tables
        except Exception as e:
            logger.error(f"Failed to list tables: {str(e)}")
            return []
    
    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific table.
        
        Args:
            table_name: Name of the table
            
        Returns:
            Dictionary with table metadata
        """
        query = f"""
        SELECT column_name, data_type, is_nullable, column_default
        FROM information_schema.columns 
        WHERE table_name = '{table_name}' AND table_schema = 'public'
        ORDER BY ordinal_position
        """
        try:
            df = self.execute(query)
            return {
                "table_name": table_name,
                "columns": df.to_dict('records'),
                "column_count": len(df)
            }
        except Exception as e:
            logger.error(f"Failed to get table info for {table_name}: {str(e)}")
            return {"table_name": table_name, "columns": [], "column_count": 0}

class LLMTool(BaseTool):
    """
    LLM tool providing language model interaction capabilities.
    
    This tool wraps the existing LLM functions from utils.llm
    and provides a standardized interface for LLM operations.
    """
    
    def __init__(self, model: str = "gpt-4o-mini", provider: str = "openai"):
        """
        Initialize LLM tool.
        
        Args:
            model: Model name to use
            provider: Provider name (openai, ollama)
        """
        super().__init__("llm", "Large Language Model interaction")
        self.model = model
        self.provider = provider
        self.llm = get_llm(model=model, provider=provider)
        logger.info(f"LLMTool initialized with model: {model} (provider: {provider})")
    
    def execute(self, prompt: str, **kwargs) -> str:
        """
        Execute LLM call synchronously.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional arguments for LLM call
            
        Returns:
            LLM response string
        """
        try:
            logger.debug(f"Executing LLM call with model {self.model}")
            response = call_llm(prompt, model=self.model, provider=self.provider, **kwargs)
            logger.info(f"LLM call completed successfully, response length: {len(response)}")
            return response
        except Exception as e:
            logger.error(f"LLM call failed: {str(e)}")
            raise
    
    async def execute_async(self, prompt: str, **kwargs) -> str:
        """
        Execute LLM call asynchronously.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional arguments for LLM call
            
        Returns:
            LLM response string
        """
        try:
            logger.debug(f"Executing async LLM call with model {self.model}")
            response = await call_llm_async(prompt, model=self.model, provider=self.provider, **kwargs)
            logger.info(f"Async LLM call completed successfully, response length: {len(response)}")
            return response
        except Exception as e:
            logger.error(f"Async LLM call failed: {str(e)}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        return {
            "model": self.model,
            "provider": self.provider,
            "tool_name": self.name
        }

class FileTool(BaseTool):
    """
    File operations tool providing file system access.
    
    This tool provides safe file operations with proper error handling
    and logging for agent file system interactions.
    """
    
    def __init__(self):
        super().__init__("file", "File operations and file system access")
        logger.info("FileTool initialized")
    
    def execute(self, operation: str, path: str, **kwargs) -> Any:
        """
        Execute file operation.
        
        Args:
            operation: Operation to perform (read, write, exists, delete)
            path: File path
            **kwargs: Additional arguments (e.g., content for write operation)
            
        Returns:
            Operation result
        """
        try:
            if operation == "read":
                return self._read_file(path, **kwargs)
            elif operation == "write":
                content = kwargs.get('content', '')
                return self._write_file(path, content, **kwargs)
            elif operation == "exists":
                return self._file_exists(path)
            elif operation == "delete":
                return self._delete_file(path)
            else:
                raise ValueError(f"Unknown operation: {operation}")
        except Exception as e:
            logger.error(f"File operation '{operation}' failed on '{path}': {str(e)}")
            raise
    
    def _read_file(self, path: str, encoding: str = 'utf-8') -> str:
        """Read file contents."""
        with open(path, 'r', encoding=encoding) as f:
            content = f.read()
        logger.debug(f"Read file: {path} ({len(content)} characters)")
        return content
    
    def _write_file(self, path: str, content: str, encoding: str = 'utf-8') -> bool:
        """Write content to file."""
        with open(path, 'w', encoding=encoding) as f:
            f.write(content)
        logger.debug(f"Wrote file: {path} ({len(content)} characters)")
        return True
    
    def _file_exists(self, path: str) -> bool:
        """Check if file exists."""
        import os
        exists = os.path.exists(path)
        logger.debug(f"File exists check: {path} -> {exists}")
        return exists
    
    def _delete_file(self, path: str) -> bool:
        """Delete file."""
        import os
        if os.path.exists(path):
            os.remove(path)
            logger.debug(f"Deleted file: {path}")
            return True
        else:
            logger.warning(f"File not found for deletion: {path}")
            return False

class ToolRegistry:
    """
    Central registry for managing tools.
    
    This registry provides a centralized way to register, retrieve,
    and execute tools across the system.
    """
    
    def __init__(self):
        """Initialize the tool registry."""
        self.tools: Dict[str, BaseTool] = {}
        self._lock = asyncio.Lock()
        logger.info("ToolRegistry initialized")
    
    def register(self, tool: BaseTool) -> None:
        """
        Register a tool in the registry.
        
        Args:
            tool: Tool instance to register
            
        Raises:
            ValueError: If tool with same name already exists
        """
        if tool.name in self.tools:
            raise ValueError(f"Tool '{tool.name}' is already registered")
        
        self.tools[tool.name] = tool
    
    def get(self, name: str) -> Optional[BaseTool]:
        """
        Get a tool by name.
        
        Args:
            name: Tool name
            
        Returns:
            Tool instance or None if not found
        """
        return self.tools.get(name)
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """List all registered tools with their metadata."""
        return [tool.to_dict() for tool in self.tools.values()]
    
    def execute(self, name: str, *args, **kwargs) -> Any:
        """
        Execute a tool by name.
        
        Args:
            name: Tool name
            *args: Tool arguments
            **kwargs: Tool keyword arguments
            
        Returns:
            Tool execution result
            
        Raises:
            ValueError: If tool not found
        """
        tool = self.get(name)
        if not tool:
            raise ValueError(f"Tool '{name}' not found")
        
        logger.debug(f"Executing tool: {name}")
        return tool(*args, **kwargs)
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics for all tools."""
        return {
            "total_tools": len(self.tools),
            "tools": [tool.get_usage_stats() for tool in self.tools.values()]
        }

# Global tool registry instance
tool_registry = ToolRegistry()

def register_default_tools():
    """
    Register default tools in the global registry.
    
    This function registers commonly used tools that are available
    by default in the system.
    """
    try:
        # Register LLM tool
        llm_tool = LLMTool()
        tool_registry.register(llm_tool)
        
        # Register file tool
        file_tool = FileTool()
        tool_registry.register(file_tool)
        
        logger.info("Default tools registered successfully")
    except Exception as e:
        logger.error(f"Failed to register default tools: {str(e)}")
        raise

def track_tool_execution(tool_name: str):
    """
    Decorator to track tool execution with timing and error handling.
    
    Args:
        tool_name: Name of the tool for logging purposes
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = datetime.now()
            try:
                result = func(*args, **kwargs)
                execution_time = (datetime.now() - start_time).total_seconds()
                logger.info(f"Tool '{tool_name}' executed successfully in {execution_time:.2f}s")
                return result
            except Exception as e:
                execution_time = (datetime.now() - start_time).total_seconds()
                logger.error(f"Tool '{tool_name}' failed after {execution_time:.2f}s: {str(e)}")
                raise
        return wrapper
    return decorator

# Initialize default tools when module is imported
register_default_tools()
