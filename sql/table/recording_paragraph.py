from sqlalchemy import Column, BigInteger, Text, DateTime, ForeignKey, Integer
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()



class RecordingParagraph(Base):
    __tablename__ = 'recording_paragraph'

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    document_id = Column(BigInteger,  nullable=False)
    paragraph_index = Column(Integer, nullable=False)  # 段落序号
    content = Column(Text, nullable=True)  # 段落文本
    embedding = Column(Text, nullable=True)  # 段落向量
    create_time = Column(DateTime, nullable=True)