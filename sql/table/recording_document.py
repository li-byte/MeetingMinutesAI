from sqlalchemy import Column, BigInteger, String, Text, DateTime, JSON, ForeignKey, Integer
from sqlalchemy.dialects.mysql import LONGTEXT
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class RecordingDocument(Base):
    __tablename__ = 'recording_document'

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    title = Column(String(512), nullable=True)
    content = Column(JSON, nullable=True)
    text_for_search = Column(LONGTEXT, nullable=True)  # 可检索文本
    summary = Column(Text, nullable=True)
    embedding = Column(Text, nullable=True)  # summary向量
    file_url = Column(String(1024), nullable=True)
    create_time = Column(DateTime, nullable=True)


