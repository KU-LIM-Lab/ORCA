# 각족 llm model이 잘 호출 되는지 확인하기 위한 모듈

# 필요한 모듈 호출
from utils.llm import call_llm, get_llm
from pathlib import Path
import json, re

def parse_llm_json(response):
    # 1) 이미 dict인 경우
    if isinstance(response, dict):
        return response
    
    # 2) 객체에 .output(str)로 들어있는 경우
    if hasattr(response, "output") and isinstance(response.output, str):
        response = response.output
    
    # 3) 문자열 전처리
    if isinstance(response, str):
        s = response.strip()

        # 3-a) 코드펜스 제거: ```json ... ``` 또는 ``` ... ```
        if s.startswith("```"):
            # 맨 앞 ```[언어]? 제거
            s = re.sub(r"^```[a-zA-Z0-9]*\s*", "", s)
            # 맨 뒤 ``` 제거
            s = re.sub(r"\s*```$", "", s).strip()

        # 3-b) 본문에 설명+JSON이 섞여 있으면, 첫 {부터 마지막 }까지만 추출
        if not s.startswith("{"):
            m = re.search(r"\{.*\}", s, flags=re.DOTALL)
            if m:
                s = m.group(0).strip()

        # 3-c) 최종적으로 JSON 시도
        return json.loads(s)

    raise ValueError("Unsupported LLM output type")

def check_llm_working(model_name, prompt, provider):
    """
    input : 모델 이름, 프롬프트
    output : 사용 모델 이름, 모델 호출 실행여부, 실행됐다면 output
    """
    try : 
        llm = get_llm(model=model_name, provider=provider)

        response = llm.invoke(prompt)

        try:
            result = parse_llm_json(response)
            print("Parsed JSON:", result)

        except json.JSONDecodeError as e:
            print("❌ Failed to parse LLM output as JSON.")
            print("Error:", e)
            print("Output was:", response)
            result = None
    
    except Exception as e:
        raise RuntimeError(f"Failed to call {model_name}: {e}")
    
if __name__ == "__main__":

    model_name = "llama3:latest"
    ddl_path = Path(f"schema/financial.sql")
            
    with open(ddl_path, "r", encoding="utf-8") as file:
        ddl = file.read()
    prompt = f"""
                Given the following database schema, recommend which tables should be examined 
                to analyze and answer the user's query.

                User query: "Which district has the most accounts, and how many accounts does it have?"
                Database schema: {ddl}

                output examples
                {{
                    "question": "show a relation between user gender and the point",
                    "recommended_tables" : ['users','point_transaction']
                }}

                Please strictly follow the output format below:
                {{
                    "question": "Which district has the most accounts, and how many accounts does it have?",
                    "recommended_tables": ["table_name1", "table_name2", "table_name3"]
                }}
                """
    provider = "ollama"
    
    check_llm_working(model_name, prompt, provider)


