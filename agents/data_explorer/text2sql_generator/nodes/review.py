from utils.llm import call_llm
from langchain_core.language_models.chat_models import BaseChatModel
from prompts.text2sql_generator_prompts import review_noresult_template, review_result_template
from prompts.text2sql_for_causal_prompts import review_noresult_template_for_causal, review_result_template_for_causal


def review_node(state, llm: BaseChatModel):
    result = state.get("result")
    error = state.get("error")
    sql = state.get("pred") or state.get("final_sql")
    query = state.get("query")
    desc_str = state.get("desc_str")
    llm_review = state.get("llm_review", None)
    evidence = state.get("evidence", "")
    mode = state.get('analysis_mode','full_pipeline')

    # if error
    if error:
        return {**state, 
                "result": "An error occurred while executing the SQL query. We tried to solve it, but we need your help. Please provide more information.",
                "send_to": "system_node"}
    
    if llm_review:
        return {**state, "send_to": "system_node"}
    
    else:
        # if no result
        if not result or (isinstance(result, str) and "no rows" in result.lower()):
            if mode == 'data_exploration':
                prompt = review_noresult_template.format(
                    query=query,
                    desc_str=desc_str,
                    sql=sql,
                    evidence=evidence
                )
            elif mode == 'full_pipeline':
                prompt = review_noresult_template_for_causal.format(
                    query=query,
                    desc_str=desc_str,
                    sql=sql,
                    evidence=evidence
                )
            print(f"Reviewing the SQL logic...")
            reply = call_llm(prompt, llm = llm)
            
            if "No" in reply:
                reason = reply.replace("No. ", "").strip()
                return {**state, 
                        "send_to": "refiner_node",
                        "llm_review": reason,
                        "try_times": 0,
                        'messages': state.get("messages", []) + [
                            {
                                "role": "reviewer",
                                "content": reply
                                }]}
            else:
                result = "The SQL was executed successfully, but returned no rows. There might be no data that matches the input conditions."
                return {**state, 
                        "send_to": "system_node",
                        "llm_review": "✅ Doublechecked the SQL logic and confirmed it is correct.",
                        "result": result,
                        'messages': state.get("messages", []) + [
                            {
                                "role": "reviewer",
                                "content": reply
                                }]}

        # if result
        if mode == 'data_exploration':
            prompt = review_result_template.format(
                query=query,
                result=str(result)[:500]
            )
        elif mode == 'full_pipeline':
            prompt = review_result_template_for_causal.format(
                query=query,
                result=str(result)[:500]
            )
       
        print(f"Reviewing the final answer...")
        reply = call_llm(prompt, llm = llm)

        if "No" in reply:
            reason = reply.replace("No. ", "").strip()
            return {**state, 
                    "send_to": "refiner_node",
                    "llm_review": reason,
                    "try_times": 0,
                    'messages': state.get("messages", []) + [
                        {
                            "role": "reviewer",
                            "content": reply
                    }]}
        else:
            return {**state, 
                    "llm_review": "✅ Doublechecked the SQL logic and confirmed it is correct.",
                    "send_to": "system_node",
                    'messages': state.get("messages", []) + [
                        {
                            "role": "reviewer",
                            "content": reply
                    }]}