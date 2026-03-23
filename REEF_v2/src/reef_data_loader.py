"""REEF data loader. Loads REEF data from PostgreSQL."""

import pandas as pd
import sys
from pathlib import Path
from typing import Optional, Dict, Any

# add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from utils.database import Database
from utils.settings import POSTGRES_CONFIG


class REEFDataLoader:
    """Loads data from the REEF PostgreSQL database."""
    
    def __init__(self, db_name: str = "reef_db", config: Optional[Dict[str, Any]] = None):
        """
        Args:
            db_name: database name
            config: DB config dict (reads from utils.settings if None)
        """
        self.db_name = db_name
        if config is None:
            self.config = POSTGRES_CONFIG
        else:
            self.config = config
        
        self.db = Database(db_type="postgresql", config=self.config)
    
    def load_table(self, table_name: str, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Load a full table.

        Args:
            table_name: table name
            limit: max rows (all rows if None)

        Returns:
            DataFrame
        """
        query = f"SELECT * FROM {table_name}"
        if limit:
            query += f" LIMIT {limit}"
        
        rows, columns = self.db.run_query(query, db_id=self.db_name)
        return pd.DataFrame(rows, columns=columns)
    
    def load_joined_data(
        self,
        tables: list,
        join_conditions: list,
        select_columns: Optional[list] = None,
        where_clause: Optional[str] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Load data by joining multiple tables.

        Args:
            tables: list of table names (first is the main table)
            join_conditions: join condition strings
            select_columns: columns to select (all if None)
            where_clause: optional WHERE clause
            limit: max rows

        Returns:
            DataFrame
        """
        if select_columns is None:
            select_columns = ["*"]
        
        query = f"SELECT {', '.join(select_columns)} FROM {tables[0]}"
        
        for i, condition in enumerate(join_conditions):
            if i + 1 < len(tables):
                query += f" JOIN {tables[i + 1]} ON {condition}"
        
        if where_clause:
            query += f" WHERE {where_clause}"
        
        if limit:
            query += f" LIMIT {limit}"
        
        rows, columns = self.db.run_query(query, db_id=self.db_name)
        return pd.DataFrame(rows, columns=columns)
    
    def load_custom_query(self, query: str) -> pd.DataFrame:
        """
        Execute a custom SQL query.

        Args:
            query: SQL query string

        Returns:
            DataFrame
        """
        rows, columns = self.db.run_query(query, db_id=self.db_name)
        return pd.DataFrame(rows, columns=columns)
    
    def get_table_columns(self, table_name: str) -> list:
        """
        Get the list of columns in a table.

        Args:
            table_name: table name

        Returns:
            list of column names
        """
        query = f"""
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_name = '{table_name}'
        ORDER BY ordinal_position
        """
        rows, columns = self.db.run_query(query, db_id=self.db_name)
        return [row[0] for row in rows]
    
    def get_all_tables(self) -> list:
        """
        Get all table names in the database.

        Returns:
            list of table names
        """
        query = """
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public' 
        AND table_type = 'BASE TABLE'
        ORDER BY table_name
        """
        rows, columns = self.db.run_query(query, db_id=self.db_name)
        return [row[0] for row in rows]

