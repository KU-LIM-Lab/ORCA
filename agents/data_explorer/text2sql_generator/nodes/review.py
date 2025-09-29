from utils.llm import call_llm
from langchain_core.language_models.chat_models import BaseChatModel
from ORCA.prompts.text2sql_generator_prompts import review_noresult_template, review_result_template


def review_node(state, llm: BaseChatModel):
    result = state.get("result")
    error = state.get("error")
    sql = state.get("pred") or state.get("final_sql")
    query = state.get("query")
    desc_str = state.get("desc_str")
    llm_review = state.get("llm_review", None)
    evidence = state.get("evidence", "")

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
            prompt = review_noresult_template.format(
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
        prompt = review_result_template.format(
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