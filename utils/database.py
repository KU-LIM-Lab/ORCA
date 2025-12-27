import os
import sqlite3
from .settings import POSTGRES_CONFIG, SQLITE_CONFIG

import copy

class Database:
    def __init__(self, db_type="postgresql", config=None):
        self.db_type = db_type.lower()

        if config:
            self.config = copy.deepcopy(config)
        elif self.db_type == "postgresql":
            self.config = copy.deepcopy(POSTGRES_CONFIG)
        elif self.db_type == "sqlite":
            self.config = copy.deepcopy(SQLITE_CONFIG)
        else:
            raise ValueError(f"Unsupported db_type: {self.db_type}")

    def _get_psycopg2(self):
        try:
            import psycopg2  # Lazy import
            return psycopg2
        except ImportError as e:
            raise ImportError(
                "psycopg2 is required for PostgreSQL operations. Install psycopg2-binary."
            ) from e

    def get_dbtype(self):
        return self.db_type
    
    def list_databases(self):
        if self.db_type == "postgresql":
            try:
                psycopg2 = self._get_psycopg2()
                conn = psycopg2.connect(
                    host=self.config["host"],
                    port=self.config["port"],
                    dbname="postgres",  # 시스템 DB
                    user=self.config["user"],
                    password=self.config["password"]
                )
                cur = conn.cursor()
                cur.execute("SELECT datname FROM pg_database WHERE datistemplate = false;")
                db_list = [row[0] for row in cur.fetchall()]
                cur.close()
                conn.close()
                return db_list
            except Exception as e:
                print(f"Failed to list PostgreSQL databases: {e}")
                return []

        elif self.db_type == "sqlite":
            base_path = self.config.get("base_dir", "data/")
            try:
                return [
                    f for f in os.listdir(base_path)
                    if f.endswith((".sqlite", ".db"))
                ]
            except Exception as e:
                print(f"Failed to list SQLite files: {e}")
                return []
        else:
            raise ValueError(f"Unsupported db_type: {self.db_type}")
        
    def get_connection(self, db_id=None):
        if self.db_type == "postgresql":
            return self.get_pg_conn(db_id)
        elif self.db_type == "sqlite":
            return self.get_sqlite_conn()
        else:
            raise ValueError(f"Unsupported db_type: {self.db_type}")

    def get_pg_conn(self, db_id=None):
        psycopg2 = self._get_psycopg2()
        return psycopg2.connect(
            host=self.config["host"],
            port=self.config["port"],
            dbname=db_id or self.config["dbname"],
            user=self.config["user"],
            password=self.config["password"]
        )

    def get_sqlite_conn(self, db_id=None):
        sqlite_path = self.config.get("sqlite_path", "")
        if db_id:
            sqlite_path = sqlite_path.replace("{db_id}", db_id)
        if not sqlite_path:
            raise ValueError("sqlite_path must be provided for SQLite DB.")
        return sqlite3.connect(sqlite_path)

    def run_query(self, sql: str, db_id: str = None):
        """Execute SQL query and log tool call event if event logger is available."""
        import time
        import hashlib
        
        # Get event logger from global metrics collector if available
        event_logger = None
        try:
            from monitoring.metrics.collector import get_metrics_collector
            collector = get_metrics_collector()
            if collector and hasattr(collector, 'event_logger'):
                event_logger = collector.event_logger
        except Exception:
            pass
        
        # Log tool call start
        start_time = time.time()
        sql_hash = hashlib.md5(sql.encode()).hexdigest()[:8]
        
        if event_logger:
            event_logger.log_tool_call_start(
                tool_name="sql_query",
                step_id="1",  # SQL queries are typically in Step 1
                metadata={
                    "sql_hash": sql_hash,
                    "sql_preview": sql[:100] if len(sql) > 100 else sql,
                    "db_id": db_id
                }
            )
        
        try:
            if self.db_type == "postgresql":
                conn = self.get_pg_conn(db_id)
            elif self.db_type == "sqlite":
                conn = self.get_sqlite_conn(db_id)
            else:
                raise ValueError(f"Unsupported db_type: {self.db_type}")

            cur = conn.cursor()
            cur.execute(sql)
            rows = cur.fetchall()
            column_names = [desc[0] for desc in cur.description]
            cur.close()
            conn.close()
            
            # Log tool call end (success)
            duration = time.time() - start_time
            if event_logger:
                event_logger.log_tool_call_end(
                    tool_name="sql_query",
                    duration=duration,
                    success=True,
                    step_id="1",
                    metadata={
                        "sql_hash": sql_hash,
                        "row_count": len(rows),
                        "column_count": len(column_names),
                        "db_id": db_id
                    }
                )
            
            # Save artifacts if artifact manager is available
            try:
                from monitoring.experiment.utils import get_artifact_manager
                import pandas as pd
                
                artifact_manager = get_artifact_manager()
                if artifact_manager:
                    # Save SQL query
                    artifact_manager.save_artifact(
                        artifact_type="sql",
                        data=sql,
                        filename="step1_final.sql",
                        step_id="1",
                        metadata={"db_id": db_id, "sql_hash": sql_hash}
                    )
                    
                    # Save dataset if data was returned
                    if rows and column_names:
                        df = pd.DataFrame(rows, columns=column_names)
                        artifact_manager.save_artifact(
                            artifact_type="dataset",
                            data=df,
                            filename="step1_dataset.parquet",
                            step_id="1",
                            metadata={"rows": len(rows), "columns": len(column_names)}
                        )
                        
                        # Save schema
                        schema_info = {
                            "columns": list(column_names),
                            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                            "shape": df.shape,
                            "null_counts": df.isnull().sum().to_dict()
                        }
                        artifact_manager.save_artifact(
                            artifact_type="schema",
                            data=schema_info,
                            filename="step1_schema.json",
                            step_id="1",
                            metadata={}
                        )
            except Exception as e:
                # Don't fail if artifact saving fails
                import logging
                logging.getLogger(__name__).warning(f"Failed to save SQL artifacts: {e}")
            
            return rows, column_names
            
        except Exception as e:
            # Log tool call end (failure)
            duration = time.time() - start_time
            if event_logger:
                event_logger.log_tool_call_end(
                    tool_name="sql_query",
                    duration=duration,
                    success=False,
                    error=str(e),
                    step_id="1",
                    metadata={
                        "sql_hash": sql_hash,
                        "db_id": db_id
                    }
                )
            raise
    
    def run_queries(self, queries: list[str], db_id: str = None) -> list:
        conn = self.get_connection(db_id)
        cursor = conn.cursor()
        results = []
        try:
            for sql in queries:
                cursor.execute(sql)
                try:
                    rows = cursor.fetchall()
                    column_names = [desc[0] for desc in cursor.description]
                    results.append((rows, column_names))
                except Exception:
                    # SELECT가 아닌 경우 (예: UPDATE, INSERT)
                    results.append((None, None))
            return results
        finally:
            cursor.close()
            conn.close()
