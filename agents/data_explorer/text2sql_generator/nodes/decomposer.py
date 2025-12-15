import re
from utils.llm import call_llm
from prompts.text2sql_generator_prompts import decompose_template
from prompts.text2sql_for_causal_prompts import decompose_template_for_causal

from langchain_core.language_models.chat_models import BaseChatModel

def decomposer_node(state, llm: BaseChatModel):
    schema_info = state['desc_str']
    fk_info = state['fk_str']
    query = state['query']
    selected_info = state.get("extracted_schema")
    evidence = state.get('evidence')
    mode = state.get('analysis_mode','full_pipeline')

    if mode == 'data_exploration':

        prompt = decompose_template.format(
            desc_str=schema_info, fk_str=fk_info, query=query, evidence=evidence
        )
    elif mode == "full_pipeline":
        prompt = decompose_template_for_causal.format(
            desc_str=schema_info, fk_str=fk_info, query=query, selected_info=selected_info, evidence=evidence
        )
    llm_reply = call_llm(prompt, llm=llm)

    all_sqls = []
    for match in re.finditer(r'```sql(.*?)```', llm_reply, re.DOTALL):
        all_sqls.append(match.group(1).strip())
    if all_sqls:
        sql = all_sqls[-1]
    else:
        raise ValueError("No SQL found in the LLM response")

    return {
        **state,
        'final_sql': sql,
        'qa_pairs': llm_reply,
        'send_to': 'refiner_node',
        'messages': state['messages'] + [
            {"role": "decomposer", "content": llm_reply}
        ]
    }