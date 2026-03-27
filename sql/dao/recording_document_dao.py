import datetime
import json

from sqlalchemy import text
from sqlalchemy.orm import Session

from sql.table.recording_document import RecordingDocument
from sql.table.recording_paragraph import RecordingParagraph


class RecordingDAO:

    @staticmethod
    def add_document_with_paragraphs(session: Session, title, content_json, embedding, summary, file_url,
                                     paragraphs: list):
        """
        插入文档及对应段落（一次事务）

        参数:
            session (Session): SQLAlchemy 会话对象
            title (str): 文档标题
            content_json (list[dict]): 文档内容 JSON
            summary (str): 文档摘要
            file_url (str): 文件 URL
            paragraphs (list[dict]): 每个段落字典包含 "content" 和 "embedding" 列表

        返回:
            int: 新增文档 ID

        异常:
            Exception: 当插入失败时抛出异常，事务会自动回滚
        """
        try:
            # 序列化文档 embedding 和内容
            doc = RecordingDocument(
                title=title,
                content=content_json,
                text_for_search=summary,
                summary=summary,
                embedding=str(embedding).replace(" ", ""),
                file_url=file_url,
                create_time=datetime.datetime.now()
            )
            session.add(doc)
            session.flush()  # 生成 doc.id，但不提交

            # 构造段落对象列表
            para_objs = []
            for idx, para in enumerate(paragraphs):
                # 验证必要字段
                if not para.get("segment"):
                    raise ValueError(f"Paragraph {idx} missing required field: segment")

                embedding_str = str(para.get("embedding", [])).replace(" ", "")
                p = RecordingParagraph(
                    document_id=doc.id,
                    paragraph_index=idx,
                    content=para.get("segment"),
                    embedding=embedding_str,
                    create_time=datetime.datetime.now()
                )
                para_objs.append(p)

            # 批量添加段落
            session.add_all(para_objs)
            session.commit()  # 一次性提交事务
            return doc.id

        except Exception as e:
            session.rollback()  # 发生异常时回滚事务
            # 可以根据需要记录日志
            # logger.error(f"Failed to add document with paragraphs: {e}")
            raise  # 重新抛出异常，让调用方处理

    @staticmethod
    def search_document_and_paragraphs(session: Session, query_vector: list, query_text: str, limit=10):
        """
        两步走搜索：
        1. 查询文档ID和textScore（按文档向量+段落最相关向量+textScore排序）
        2. 根据ID查询完整文档及段落内容
        返回结果保留原字段
        """
        query_vector_str = str(query_vector).replace(" ", "")

        # =======================
        # Step 1: 查询文档ID和textScore
        # =======================

        sql_doc_ids = text("""
                           WITH paragraph_distance AS (SELECT document_id,
                                                              MIN(l2_distance(embedding, :query_vector_str)) AS min_paragraph_distance
                                                       FROM recording_paragraph
                                                       WHERE embedding IS NOT NULL
                                                       GROUP BY document_id)
                           SELECT d.id AS                          document_id,
                                  -- 直接加权评分：向量距离 + 段落距离 - 文本得分
                                  ((l2_distance(d.embedding, :query_vector_str) +
                                    COALESCE(p.min_paragraph_distance, 0)) * :vector_weight
                                       - MATCH
                                   (d.text_for_search)             AGAINST(:query_text IN NATURAL LANGUAGE MODE) * :text_weight ) AS final_score
                           FROM recording_document d
                               LEFT JOIN paragraph_distance p
                           ON d.id = p.document_id
                           ORDER BY final_score ASC
                               LIMIT :limit
                           """)

        doc_id_results = session.execute(sql_doc_ids, {
            "query_vector_str": query_vector_str,
            "query_text": query_text,
            "limit": limit,
            "vector_weight": 1.0,
            "text_weight": 2.0
        }).fetchall()

        doc_ids = [row.document_id for row in doc_id_results]
        text_scores_map = {row.document_id: row.final_score for row in doc_id_results}

        if not doc_ids:
            return []

        # =======================
        # Step 2: 查询完整文档和段落
        # =======================
        from sqlalchemy.orm import aliased

        documents = session.query(RecordingDocument).filter(RecordingDocument.id.in_(doc_ids)).all()
        results = []

        for doc in documents:
            paragraphs = session.query(RecordingParagraph) \
                .filter_by(document_id=doc.id) \
                .order_by(RecordingParagraph.paragraph_index).all()

            para_list = [{
                "id": p.id,
                "paragraph_index": p.paragraph_index,
                "content": p.content
            } for p in paragraphs]

            results.append({
                "id": doc.id,
                "title": doc.title,
                "summary": doc.summary,
                "content": doc.content,
                "textScore": min(text_scores_map.get(doc.id, 0), 1)/1,
                "paragraphs": para_list
            })

        # 按第一步的排序顺序返回
        results.sort(key=lambda x: doc_ids.index(x["id"]))
        return results
