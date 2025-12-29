import json, os, yaml, argparse, decimal
import time, traceback
from pathlib import Path
from datetime import datetime
from argparse import Namespace

from utils.prettify import print_final_output_explorer, print_final_output_recommender, print_final_output_causal
from agents.table_explorer import generate_description_graph
from agents.table_recommender import generate_table_recommendation_graph
from agents.text2sql_generator import generate_text2sql_graph 
from agents.causal_analysis import generate_causal_analysis_graph
from main_agent import Agent
from langchain_core.messages import HumanMessage


# ----- CONFIGURATION -----
CONFIG_PATH = "_config.yml"

def dict2namespace(config):
    ns = argparse.Namespace()
    for k, v in config.items():
        setattr(ns, k, dict2namespace(v) if isinstance(v, dict) else v)
    return ns

with open(CONFIG_PATH, "r") as f:
    config = dict2namespace(yaml.safe_load(f))


# ----- CUSTOM JSON ENCODER -----
class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        try:
            return str(obj)
        except Exception:
            return super().default(obj)

# ----- HELPER -----
def load_questions(path: Path) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_results(path: Path, results: list):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, cls=EnhancedJSONEncoder)
    # 중간 결과 (.jsonl) 삭제
    jsonl_path = path.with_suffix(".jsonl")
    if jsonl_path.exists():
        jsonl_path.unlink()

def save_checkpoint(result: dict, result_dir: Path, prefix: str, data: str, timestamp: str):
    file_path = result_dir / f"{prefix}_{data}_{timestamp}.jsonl"
    with open(file_path, "a", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, cls=EnhancedJSONEncoder)
        f.write("\n")


# ----- EXPERIMENT FUNCTIONS -----
def experiment_tabledescription(data_dir: Path, result_dir: Path, timestamp: str, data: str, llm=None):
    questions = load_questions(data_dir / "table_explorer.json")
    results = []

    for i, q in enumerate(questions):
        print(f"processing {i+1}/{len(questions)}")
        app = generate_description_graph(llm=llm)
        result = app.invoke({"input": q['table_name'], "db_id": q.get('db_id')})
        final_output = print_final_output_explorer(result["final_output"])

        result_entry = {
            "table_name": q['table_name'],
            "final_output": final_output,
            "components": {
                "table_analysis": result["table_analysis"],
                "recommended_analysis": str(result["recommended_analysis"]),
                "related_tables": result["related_tables"]
            }
        }
        results.append(result_entry)
        save_checkpoint(result_entry, result_dir, "table_explorer", data, timestamp)

    save_results(result_dir / f"table_explorer_{data}_{timestamp}.json", results)
    print(f"\n✅ Table Description experiment completed. Results saved to {result_dir / f'table_explorer_{data}_{timestamp}.json'}")


def experiment_recommendtable(data_dir: Path, result_dir: Path, timestamp: str, data: str, llm=None):
    questions = load_questions(data_dir / "table_recommender.json")
    results = []

    for i, q in enumerate(questions):
        print(f"processing {i+1}/{len(questions)}")
        app = generate_table_recommendation_graph(llm=llm)
        result = app.invoke({"input": q['question'], "db_id": q.get('db_id')})
        final_output = print_final_output_recommender(result["final_output"])

        result_entry = {
            "question": q['question'],
            "final_output": final_output,
            "components": {
                "objective_summary": result["objective_summary"],
                "recommended_tables": str(result["recommended_tables"]),
                "recommended_method": result["recommended_method"],
                "erd_image_path": result["erd_image_path"]
            }
        }
        results.append(result_entry)
        save_checkpoint(result_entry, result_dir, "table_recommender", data, timestamp)

    save_results(result_dir / f"table_recommender_{data}_{timestamp}.json", results)
    print(f"\n✅ Table Recommendation experiment completed. Results saved to {result_dir / f'table_recommender_{data}_{timestamp}.json'}")


def experiment_text2sql_generator(data_dir: Path, result_dir: Path, timestamp: str, data: str, llm=None):
    questions = load_questions(data_dir / "text2sql_generator.json")
    results = []

    for i, q in enumerate(questions):
        print(f"\n[{i+1}/{len(questions)}] Processing Q{q['question_id']}: {q['question']}")
        app = generate_text2sql_graph(llm=llm)

        state = {
            "query": q["question"],
            "evidence": q.get("evidence", ""),
            "messages": [{"role": "user", "content": q["question"]}],
            "desc_str": None,
            "fk_str": None,
            "extracted_schema": None,
            "final_sql": None,
            "qa_pairs": None,
            "pred": None,
            "result": None,
            "error": None,
            "pruned": False,
            "send_to": "selector_node",
            "try_times": 0,
            "llm_review": None,
            "review_count": 0,
            "output": None,
            "db_id": q["db_id"],
            "notes": None
        }

        try:
            result = app.invoke(state)
            output = result.get("output", {})
            messages = result.get("messages", []) + [{"role": "system", "content": output}]

            result_entry = {
                "question_id": q["question_id"],
                "question": q["question"],
                "ground_truth_sql": q["SQL"],
                "generated_sql": output.get("sql"),
                "evidence": q.get("evidence", ""),
                "difficulty": q.get("difficulty", ""),
                "db_id": q["db_id"],
                "error": output.get("error"),
                "logs": messages
            }

        except Exception as e:
            print("❌ Error during execution:", e)
            result_entry = {
                "question_id": q["question_id"],
                "question": q["question"],
                "ground_truth_sql": q["SQL"],
                "generated_sql": None,
                "evidence": q.get("evidence", ""),
                "difficulty": q.get("difficulty", ""),
                "db_id": q["db_id"],
                "error": str(e),
                "logs": []
            }

        results.append(result_entry)
        save_checkpoint(result_entry, result_dir, "text2sql_generator", data, timestamp)

    save_results(result_dir / f"text2sql_generator_{data}_{timestamp}.json", results)
    print(f"\n✅ Text2sql_generator experiment completed. Results saved to {result_dir / f'text2sql_generator_{data}_{timestamp}.json'}")

    
def experiment_mainagent(data_dir: Path, result_dir: Path, timestamp: str, data: str, llm=None):
    questions = load_questions(data_dir / "main_agent.json")
    results = []
    duration = []

    for i, q in enumerate(questions):
        print(f"\n[{i+1}/{len(questions)}] Input: {q['input']}")
        agent = Agent(config, prompt_routing_path="prompts/routing.txt", db_id=q.get("db_id", "daa"))

        try:
            start = time.time()
            output = agent.execute(q["input"])
            end = time.time() 
            duration.append(end - start)

            messages = agent.memory.chat_memory.messages
            result_entry = {
                "input": q["input"],
                "output": output,
                "conversation": [
                    {"role": "user" if isinstance(m, HumanMessage) else "assistant", "content": m.content}
                    for m in messages
                ]
            }

        except Exception as e:
            result_entry = {
                "input": q["input"],
                "output": None,
                "db_id": q.get("db_id", "daa"),
                "error": str(e)
            }

        results.append(result_entry)
        save_checkpoint(result_entry, result_dir, "mainagent", data, timestamp)
    
    if duration:
        avg_time = sum(duration) / len(duration)
        print(f"\n⏱️ 평균 질의 처리 시간: {avg_time:.2f}초")
        duration_entry = {
            "duration_list" : duration,
            "avg_duration" : avg_time         
            }
        results.append(duration_entry)
    else:
        print("\n⚠️ 성공적으로 처리된 질의가 없습니다.")

    save_results(result_dir / f"mainagent_{data}_{timestamp}.json", results)
    print(f"\n✅ MainAgent experiment completed. Results saved to {result_dir / f'mainagent_{data}_{timestamp}.json'}")


# ----- MAIN -----
def main():
    parser = argparse.ArgumentParser(description="Run selected experiments on a dataset")
    parser.add_argument("-d", "--data", type=str, choices=["daa", "bird"], default="daa",
                        help="Dataset name: 'daa' or 'bird'")
    parser.add_argument("-t", "--task", type=str, nargs="+", choices=[
        "table_explorer", "table_recommender", "text2sql_generator", "main_agent"
    ], default=["table_explorer", "table_recommender", "text2sql_generator"],
    help="Tasks to run. Multiple allowed. Choose from: table_explorer, table_recommender, text2sql_generator, main_agent")

    args = parser.parse_args()
    data = args.data
    tasks = set(args.task)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_dir = Path(f"experiments/questions/{data}")
    result_dir = Path(f"experiments/results/{data}")
    os.makedirs(result_dir, exist_ok=True)

    if "table_explorer" in tasks:
        experiment_tabledescription(data_dir, result_dir, timestamp, data)
    if "table_recommender" in tasks:
        experiment_recommendtable(data_dir, result_dir, timestamp, data)
    if "text2sql_generator" in tasks:
        experiment_text2sql_generator(data_dir, result_dir, timestamp, data)
    # if "causal_analysis" in tasks:
    #     experiment_causalanalysis(data_dir, result_dir, timestamp, data)
    if "main_agent" in tasks:
        experiment_mainagent(data_dir, result_dir, timestamp, data)

if __name__ == "__main__":
    main()