from dataclasses import dataclass, asdict
import json


from dataclasses import dataclass, asdict
import json
from typing import List

@dataclass
class SummaryEntity:
    """
    总结内容实体类
    - summary: 存储最终压缩总结
    - segments: 存储按语义分段的原文内容
    """
    summary: str
    segments: List[str] = None

    @staticmethod
    def from_llm_json(llm_json_str: str):
        """
        从大模型返回的 JSON 字符串解析出 summary 和 segments，生成 SummaryEntity 实例
        :param llm_json_str: 大模型返回的 JSON 字符串，如
               '{"summary":"...总结内容...","segments":["段落1","段落2"]}'
        :return: SummaryEntity 实例
        """
        try:
            data = json.loads(llm_json_str)
            summary_text = data.get("summary", "")
            segments_list = data.get("segments", [])
            # 确保 segments 是列表
            if not isinstance(segments_list, list):
                segments_list = []
            return SummaryEntity(summary=summary_text, segments=segments_list)
        except json.JSONDecodeError:
            # JSON 无效时，返回空总结
            return SummaryEntity(summary="", segments=[])

    def to_dict(self) -> dict:
        """转换为字典"""
        return asdict(self)

    def to_json(self, indent: int = 4) -> str:
        """转换为 JSON 字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


def build_lecture_summary_prompt(transcribed_text: str) -> str:
    """
    构建讲课/培训类音频总结提示词

    :param transcribed_text: 音频转写文本
    :return: 完整 prompt
    """

    system_prompt = (
        "你是一个专业的课程内容总结助手，擅长整理讲课、培训、知识分享类音频内容。"
        "你的目标是在保证信息完整性的前提下，对内容进行结构化总结。"

        "总结要求："
        "1. 按知识结构组织内容（如章节、主题、小节）。"
        "2. 提取核心知识点、概念定义、方法步骤。"
        "3. 保留关键解释、举例和结论。"
        "4. 如果存在因果关系或逻辑推导，请明确表达。"
        "5. 去除口语冗余，但保留有效信息。"
        "6. 优先保证信息完整，而不是过度简化。"

        "输出格式要求："
        "1. 总结内容使用 Markdown 格式（如标题、列表）。"
        "2. 结构清晰，层级分明。"
        "3. 最终输出必须严格为 JSON 格式："
        "{\"text\":\"...总结内容...\"}"
        "4. 不要添加任何额外解释或前缀。"
    )

    user_prompt = (
        "以下是讲课类音频转写内容，请进行结构化总结：\n\n"
        f"{transcribed_text}"
    )

    return f"{system_prompt}\n\n{user_prompt}"

def build_dialogue_summary_prompt4(transcribed_text: str) -> str:
    """
    构建对话类音频总结提示词（总结 + 原文分段）

    :param transcribed_text: 音频转写文本
    :return: 完整 prompt
    """

    system_prompt = (
        "你是一个专业的对话内容总结助手，擅长对多人对话、访谈、会议内容进行压缩总结，并按照语义分段。"

        "你的任务包含两个部分："
        "1）生成高度精简的整体总结（保留议题、核心观点、结论，忽略细节、例子、重复）"
        "2）按语义主题将原文内容合并为段落（同一语义主题的连续句子合并为一个段落）"

        "【分段要求】"
        "1. 根据语义主题合并原文内容，同一主题的连续句子合并为一个段落。"
        "2. 合并时不修改原文内容，保留原文的句子和表达方式。"
        "3. 每个段落表达一个完整的语义单元（如一个议题、一个观点、一个子话题）。"
        "4. 分段数量根据内容自动生成，保证语义清晰、段落边界合理。"

        "【总结要求】"
        "1. 仅保留最关键的信息（议题、核心观点、结论）。"
        "2. 控制在3-6条以内，每条一句话，尽量简短。"
        "3. 使用 Markdown 列表（- 开头）。"
        "4. 不展开解释或添加例子。"

        "【输出格式要求】"
        "必须严格输出 JSON，不包含任何额外说明："
        "{"
        "\"summary\": \"- 精简总结1\\n- 精简总结2\","
        "\"segments\": ["
        "  \"合并后的段落1\","
        "  \"合并后的段落2\""
        "]"
        "}"
        "确保 JSON 合法可解析。"
    )

    user_prompt = (
        "以下是对话类音频转写内容，请生成 summary 和按语义合并后的原文段落：\n\n"
        f"{transcribed_text}"
    )

    return f"{system_prompt}\n\n{user_prompt}"

def build_dialogue_summary_prompt(transcribed_text: str) -> str:
    """
    构建对话/访谈类音频总结提示词

    :param transcribed_text: 音频转写文本
    :return: 完整 prompt
    """

    system_prompt = (
        "你是一个专业的对话内容分析助手，擅长整理多人对话、访谈、会议记录等音频内容。"
        "你的目标是在保证信息完整性的前提下，提取关键讨论内容和结论。"

        "总结要求："
        "1. 区分不同发言人的观点（如有可能）。"
        "2. 提取每个参与者的核心观点和立场。"
        "3. 总结讨论的关键议题。"
        "4. 提取达成的结论、决策或共识。"
        "5. 如果存在分歧，请明确列出不同意见。"
        "6. 保留关键细节（时间、数据、行动项等）。"
        "7. 去除寒暄和无效对话，但不要丢失有效信息。"
        "8. 优先保证信息完整，而不是过度压缩。"

        "输出格式要求："
        "1. 总结内容使用 Markdown 格式（如分点、分段）。"
        "2. 建议包含：议题、观点、结论、行动项等结构。"
        "3. 最终输出必须严格为 JSON 格式："
        "{\"text\":\"...总结内容...\"}"
        "4. 不要添加任何额外解释或前缀。"
    )

    user_prompt = (
        "以下是对话类音频转写内容，请进行总结：\n\n"
        f"{transcribed_text}"
    )

    return f"{system_prompt}\n\n{user_prompt}"


def build_summary_prompt(param, prompt_type: int) -> str | None:
    if prompt_type == 1:
        # 构建对话/访谈类音频总结提示词
        return build_lecture_summary_prompt(param)
    elif prompt_type == 2:
        # 构建讲课/培训类音频总结提示词
        return build_dialogue_summary_prompt4(param)
    return None
