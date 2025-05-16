from typing import Dict, List, Any, Optional
import pymysql
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL
from sqlalchemy.exc import SQLAlchemyError
import logging

logger = logging.getLogger(__name__)

class DatabaseConnection:
    """管理MySQL数据库连接和查询"""
    
    def __init__(self, db_url: str):
        """
        初始化数据库连接
        
        Args:
            db_url: 数据库连接URL，格式: mysql+pymysql://user:password@host:port/dbname
        """
        self.db_url = db_url
        self.engine = None
        self._connect()
    
    def _connect(self):
        """建立数据库连接"""
        try:
            self.engine = create_engine(self.db_url, pool_recycle=3600)
            # 测试连接
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("数据库连接成功")
        except SQLAlchemyError as e:
            logger.error(f"数据库连接失败: {e}")
            raise
    
    def execute_query(self, query: str, params: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        执行SQL查询并返回结果列表
        
        Args:
            query: SQL查询语句
            params: 查询参数
            
        Returns:
            查询结果列表，每个元素为一行数据的字典
        """
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(query), params or {})
                columns = result.keys()
                return [dict(zip(columns, row)) for row in result.fetchall()]
        except SQLAlchemyError as e:
            logger.error(f"查询执行失败: {e}")
            raise
    
    def get_table_columns(self, table_name: str) -> List[str]:
        """
        获取表的所有列名
        
        Args:
            table_name: 表名
            
        Returns:
            列名列表
        """
        query = f"""
            SELECT COLUMN_NAME 
            FROM INFORMATION_SCHEMA.COLUMNS 
            WHERE TABLE_SCHEMA = DATABASE() 
            AND TABLE_NAME = :table_name
        """
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(query), {"table_name": table_name})
                return [row[0] for row in result.fetchall()]
        except SQLAlchemyError as e:
            logger.error(f"获取表列信息失败: {e}")
            raise
    
    def close(self):
        """关闭数据库连接"""
        if self.engine:
            self.engine.dispose()
            logger.info("数据库连接已关闭")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def create_db_url(username: str, password: str, host: str, port: int, database: str) -> str:
    """
    创建数据库连接URL
    
    Args:
        username: 数据库用户名
        password: 数据库密码
        host: 数据库主机
        port: 数据库端口
        database: 数据库名
        
    Returns:
        格式化的数据库连接URL
    """
    return f"mysql+pymysql://{username}:{password}@{host}:{port}/{database}"
