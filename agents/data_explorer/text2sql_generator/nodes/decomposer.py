import re
from utils.llm import call_llm
from ORCA.prompts.text2sql_generator_prompts import decompose_template

from langchain_core.language_models.chat_models import BaseChatModel

def decomposer_node(state, llm: BaseChatModel):
    schema_info = state['desc_str']
    fk_info = state['fk_str']
    query = state['query']
    evidence = state.get('evidence')

    prompt = decompose_template.format(
        desc_str=schema_info, fk_str=fk_info, query=query, evidence=evidence
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