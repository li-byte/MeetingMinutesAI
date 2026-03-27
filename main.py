from typing import List, Dict

from sql.dao.recording_document_dao import RecordingDAO
from sql.db import DBSession
from audioAnalysis.analysis_tool import process_media
from tool.embedding import get_embedding
from tool.large_model import summary_analysis


def extract_text_from_transcripts(transcripts: List[Dict]) -> List[str]:
    """
    将转写结果列表提取为纯文本列表，只保留 text 字段。

    :param transcripts: [{"start":..., "end":..., "speaker":..., "text":...}, ...]
    :return: ["文本1", "文本2", ...]
    """
    text_list = [entry.get("text", "").strip() for entry in transcripts if "text" in entry]
    return text_list


# if __name__ == '__main__':
#     DBSession.init()
#     session = DBSession.get_session()
#
#     q = "医药"
#     d = RecordingDAO.search_document_and_paragraphs(session, get_embedding(q), q, 100)
#     print(json.dumps(d, ensure_ascii=False, indent=4))


import time

if __name__ == '__main__':
    title = "医药NLP项目"
    url = "x.mp3"

    # 记录媒体文件处理时间
    start = time.time()
    data = process_media(url)
    end = time.time()
    print(f"媒体文件处理执行时间: {end - start:.2f} 秒")

    # 记录文本提取执行时间
    start = time.time()
    list_d = extract_text_from_transcripts(data)
    end = time.time()
    print(f"文本提取执行时间: {end - start:.2f} 秒")

    # 记录摘要分析执行时间
    start = time.time()
    future_summary = summary_analysis(list_d, 2)
    end = time.time()
    print(f"摘要分析执行时间: {end - start:.2f} 秒")

    # 记录摘要向量化执行时间
    start = time.time()
    summary_embedding = get_embedding(future_summary.summary)
    end = time.time()
    print(f"摘要向量化执行时间: {end - start:.2f} 秒")

    # 记录分段向量化执行时间
    start = time.time()
    list_d = []
    for segment in future_summary.segments:
        seg_embedding = get_embedding(segment)
        list_d.append({"embedding": seg_embedding, "segment": segment})
    end = time.time()
    print(f"分段向量化执行时间: {end - start:.2f} 秒")

    # 记录数据库操作总时间
    start = time.time()
    DBSession.init()
    session = DBSession.get_session()
    RecordingDAO.add_document_with_paragraphs(session, title=title, content_json=data, embedding=summary_embedding,
                                              summary=future_summary.summary, file_url=url, paragraphs=list_d)
    end = time.time()
    print(f"数据库操作执行时间: {end - start:.2f} 秒")

    print(data)









