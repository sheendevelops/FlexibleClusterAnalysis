import os
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from contextlib import contextmanager
import psycopg2
from psycopg2.extras import RealDictCursor, Json

class DatabaseHelper:
    """
    Helper class for database operations with PostgreSQL
    """
    
    def __init__(self):
        self.database_url = os.environ.get('DATABASE_URL')
        if not self.database_url:
            raise ValueError("DATABASE_URL environment variable not set")
    
    @contextmanager
    def get_connection(self):
        """Get a database connection with context manager"""
        conn = None
        try:
            conn = psycopg2.connect(self.database_url)
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            raise e
        finally:
            if conn:
                conn.close()
    
    def execute_query(self, query: str, params: tuple = None, fetch: bool = False) -> Optional[List[Dict]]:
        """Execute a query and optionally fetch results"""
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(query, params)
                if fetch:
                    return [dict(row) for row in cursor.fetchall()]
                conn.commit()
                return None
    
    def insert_one(self, table: str, data: Dict[str, Any]) -> int:
        """Insert a single record and return the ID"""
        columns = list(data.keys())
        placeholders = ', '.join(['%s'] * len(columns))
        columns_str = ', '.join(columns)
        
        query = f"""
        INSERT INTO {table} ({columns_str}) 
        VALUES ({placeholders}) 
        RETURNING id
        """
        
        values = []
        for value in data.values():
            if isinstance(value, (dict, list)):
                values.append(Json(value))
            else:
                values.append(value)
        
        result = self.execute_query(query, tuple(values), fetch=True)
        return result[0]['id'] if result else None
    
    def update_one(self, table: str, record_id: int, data: Dict[str, Any]) -> bool:
        """Update a single record by ID"""
        set_clauses = []
        values = []
        
        for key, value in data.items():
            set_clauses.append(f"{key} = %s")
            if isinstance(value, (dict, list)):
                values.append(Json(value))
            else:
                values.append(value)
        
        set_clause = ', '.join(set_clauses)
        query = f"UPDATE {table} SET {set_clause} WHERE id = %s"
        values.append(record_id)
        
        self.execute_query(query, tuple(values))
        return True
    
    def find_one(self, table: str, conditions: Dict[str, Any]) -> Optional[Dict]:
        """Find a single record by conditions"""
        where_clauses = []
        values = []
        
        for key, value in conditions.items():
            where_clauses.append(f"{key} = %s")
            values.append(value)
        
        where_clause = ' AND '.join(where_clauses)
        query = f"SELECT * FROM {table} WHERE {where_clause} LIMIT 1"
        
        result = self.execute_query(query, tuple(values), fetch=True)
        return result[0] if result else None
    
    def find_many(self, table: str, conditions: Dict[str, Any] = None, limit: int = None, order_by: str = None) -> List[Dict]:
        """Find multiple records by conditions"""
        query = f"SELECT * FROM {table}"
        values = []
        
        if conditions:
            where_clauses = []
            for key, value in conditions.items():
                where_clauses.append(f"{key} = %s")
                values.append(value)
            query += f" WHERE {' AND '.join(where_clauses)}"
        
        if order_by:
            query += f" ORDER BY {order_by}"
        
        if limit:
            query += f" LIMIT {limit}"
        
        result = self.execute_query(query, tuple(values), fetch=True)
        return result or []
    
    def delete_one(self, table: str, record_id: int) -> bool:
        """Delete a record by ID"""
        query = f"DELETE FROM {table} WHERE id = %s"
        self.execute_query(query, (record_id,))
        return True
    
    def count_records(self, table: str, conditions: Dict[str, Any] = None) -> int:
        """Count records in a table"""
        query = f"SELECT COUNT(*) as count FROM {table}"
        values = []
        
        if conditions:
            where_clauses = []
            for key, value in conditions.items():
                where_clauses.append(f"{key} = %s")
                values.append(value)
            query += f" WHERE {' AND '.join(where_clauses)}"
        
        result = self.execute_query(query, tuple(values), fetch=True)
        return result[0]['count'] if result else 0
    
    def search_text(self, table: str, text_column: str, search_term: str, limit: int = 10) -> List[Dict]:
        """Search for text in a specific column"""
        query = f"""
        SELECT * FROM {table} 
        WHERE {text_column} ILIKE %s 
        ORDER BY id DESC 
        LIMIT %s
        """
        
        search_pattern = f"%{search_term}%"
        result = self.execute_query(query, (search_pattern, limit), fetch=True)
        return result or []
    
    def get_recent_records(self, table: str, limit: int = 10, timestamp_column: str = 'timestamp') -> List[Dict]:
        """Get the most recent records from a table"""
        query = f"""
        SELECT * FROM {table} 
        ORDER BY {timestamp_column} DESC 
        LIMIT %s
        """
        
        result = self.execute_query(query, (limit,), fetch=True)
        return result or []
    
    def cleanup_old_records(self, table: str, days: int = 30, timestamp_column: str = 'timestamp') -> int:
        """Remove records older than specified days"""
        query = f"""
        DELETE FROM {table} 
        WHERE {timestamp_column} < NOW() - INTERVAL '%s days'
        """
        
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, (days,))
                deleted_count = cursor.rowcount
                conn.commit()
                return deleted_count

# Global database helper instance
db = DatabaseHelper()