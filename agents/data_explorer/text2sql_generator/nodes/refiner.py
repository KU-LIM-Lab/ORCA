import re
from ORCA.prompts.text2sql_generator_prompts import refiner_template, refiner_feedback_template
from utils.llm import call_llm
from utils.database import Database
from langchain_core.language_models.chat_models import BaseChatModel

database = Database()

def refiner_node(state, llm: BaseChatModel):
    db_id = state['db_id']
    sql = state.get('pred') or state.get('final_sql')
    llm_review = state.get('llm_review')
    try_times = state.get('try_times', 0)

    if llm_review and try_times == 0:
        error_info = state.get('error', {})
        
        print("Refining SQL with feedback...")
        prompt = refiner_feedback_template.format(
            query=state['query'],
            evidence=state.get('evidence'),
            desc_str=state['desc_str'],
            fk_str=state['fk_str'],
            sql=state['final_sql'],
            review_feedback=llm_review
        )
        
        llm_reply = call_llm(prompt, llm=llm)
        all_sqls = []
        for match in re.finditer(r'```sql(.*?)```', llm_reply, re.DOTALL):
            all_sqls.append(match.group(1).strip())
        if all_sqls:
            new_sql = all_sqls[-1]
        else:
            raise ValueError("No SQL found in the LLM response")
        
        return {
            **state,
            'pred': new_sql,
            'try_times': try_times + 1,
            'send_to': 'refiner_node',
            # 'llm_review': None  # 한 번 반영했으니 초기화
            'messages': state['messages'] + [
                {"role": "refiner", "content": llm_reply}
            ]
        }

    else:
        try:
            print("Executing SQL....")
            result, columns = database.run_query(sql=sql, db_id=db_id)
            if result and len(result) > 0:
                return {
                    **state,
                    'result': result,
                    'columns': columns,  # SQL 실행 후 컬럼명 저장
                    'error': None,  # 에러 초기화
                    'send_to': 'review_node'
                }
            else: # 실행은 성공했지만 반환된 결과가 없는 경우
                return {
                    **state,
                    'result': "Sql executed but no rows returned, there might be something wrong with the sql or no data in the table that matches the query.",
                    'error': None,  # 에러 초기화
                    'send_to': 'review_node'}
                
        except Exception as e:
            print("Error executing SQL:", e)
            error_info = {'sql': sql, 'error': str(e), 'exception_class': type(e).__name__}

        if try_times >= 3:
            return {
                **state,
                'error': error_info['error'],
                'send_to': 'review_node'
            }
    
        prompt = refiner_template.format(
            query=state['query'],
            evidence=state.get('evidence'),
            desc_str=state['desc_str'],
            fk_str=state['fk_str'],
            sql=error_info['sql'],
            sql_error=error_info.get('error', ''),
            exception_class=error_info.get('exception_class', '')
        )

        llm_reply = call_llm(prompt)
        all_sqls = []
        for match in re.finditer(r'```sql(.*?)```', llm_reply, re.DOTALL):
            all_sqls.append(match.group(1).strip())
        if all_sqls:
            new_sql = all_sqls[-1]
        else:
            raise ValueError("No SQL found in the LLM response")

        return {
            **state,
            'pred': new_sql,
            'try_times': try_times + 1,
            'send_to': 'refiner_node',
            # 'llm_review': None  # 한 번 반영했으니 초기화
            'messages': state['messages'] + [
                {"role": "refiner", "content": llm_reply}
            ]
        }