from typing import List, Dict

from modelclient.deepseek_single import DeepSeekClient
from prompt import recording_analysis_prompt
from prompt.recording_analysis_prompt import SummaryEntity







def summary_analysis(list_data: list, prompt_type: int) -> SummaryEntity:
    client = DeepSeekClient()
    str_q = recording_analysis_prompt.build_summary_prompt(str(list_data), prompt_type)
    d = client.complete_text(str_q)
    summary_entity = SummaryEntity.from_llm_json(d)
    return summary_entity
