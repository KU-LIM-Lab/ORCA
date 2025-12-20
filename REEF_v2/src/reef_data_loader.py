"""
REEF 데이터 로더
PostgreSQL에서 REEF 데이터를 로드합니다.
"""

import pandas as pd
import sys
from pathlib import Path
from typing import Optional, Dict, Any

# 프로젝트 루트를 path에 추가
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from utils.database import Database
from utils.settings import POSTGRES_CONFIG


class REEFDataLoader:
    """REEF 데이터베이스에서 데이터를 로드하는 클래스"""
    
    def __init__(self, db_name: str = "reef_db", config: Optional[Dict[str, Any]] = None):
        """
        Args:
            db_name: 데이터베이스 이름
            config: 데이터베이스 설정 (None이면 utils.settings에서 가져옴)
        """
        self.db_name = db_name
        if config is None:
            self.config = POSTGRES_CONFIG
        else:
            self.config = config
        
        self.db = Database(db_type="postgresql", config=self.config)
    
    def load_table(self, table_name: str, limit: Optional[int] = None) -> pd.DataFrame:
        """
        테이블 전체를 로드합니다.
        
        Args:
            table_name: 테이블 이름
            limit: 최대 행 수 (None이면 전체)
        
        Returns:
            데이터프레임
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
        여러 테이블을 조인하여 데이터를 로드합니다.
        
        Args:
            tables: 테이블 이름 리스트 (첫 번째가 메인 테이블)
            join_conditions: 조인 조건 리스트 (예: ["orders.user_id = users.user_id"])
            select_columns: 선택할 컬럼 리스트 (None이면 모든 컬럼)
            where_clause: WHERE 절 (예: "orders.total_amount > 100")
            limit: 최대 행 수
        
        Returns:
            데이터프레임
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
        커스텀 SQL 쿼리를 실행합니다.
        
        Args:
            query: SQL 쿼리 문자열
        
        Returns:
            데이터프레임
        """
        rows, columns = self.db.run_query(query, db_id=self.db_name)
        return pd.DataFrame(rows, columns=columns)
    
    def get_table_columns(self, table_name: str) -> list:
        """
        테이블의 컬럼 목록을 가져옵니다.
        
        Args:
            table_name: 테이블 이름
        
        Returns:
            컬럼 이름 리스트
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
        데이터베이스의 모든 테이블 목록을 가져옵니다.
        
        Returns:
            테이블 이름 리스트
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

