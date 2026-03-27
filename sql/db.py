# db_connection.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session

class DBSession:
    """数据库连接和 Session 管理"""

    _engine = None
    _SessionFactory = None

    @classmethod
    def init(cls):
        # 对应你 Spring 配置的参数
        user = 'root'
        password = ''  # <== 替换成实际密码
        host = '127.0.0.1'
        port = 2881
        database = 'test'

        url = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}?charset=utf8mb4"

        cls._engine = create_engine(
            url,
            pool_size=10,
            max_overflow=20,
            pool_recycle=3600,
            echo=False  # 打印 SQL 可设置 True
        )
        cls._SessionFactory = scoped_session(sessionmaker(bind=cls._engine))

    @classmethod
    def get_session(cls):
        if cls._SessionFactory is None:
            raise RuntimeError("DBSession not initialized. Call DBSession.init() first.")
        return cls._SessionFactory()