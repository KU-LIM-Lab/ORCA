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
        return rows, column_names
    
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
