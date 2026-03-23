import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # ORCA/ root

import ast, json, argparse

from utils.database import Database
from utils.llm import get_llm
from prompts.grading_prompts import table_description_grading_prompt


# ----- ORCA FORMAT NORMALIZATION -----

def _parse_orca_recommended_tables(raw) -> list:
    """Parse the string-repr list stored by run_experiments.py into a plain list."""
    if isinstance(raw, list):
        return raw
    if not raw:
        return []
    try:
        result = ast.literal_eval(str(raw))
        return result if isinstance(result, list) else []
    except (ValueError, SyntaxError):
        return []


def normalize_orca_result(r: dict, task: str, ground_truth_map: dict = None) -> dict:
    """
    Map ORCA agent result fields to the flat format expected by the eval functions.

    table_explorer  : final_output (formatted str) → description
    table_recommender: components.recommended_tables (str repr) → predicted_tables
                       + ground_truth injected from question file
    text2sql        : already compatible, no changes needed
    """
    r = dict(r)  # shallow copy — don't mutate original

    if task == "table_explorer":
        r.setdefault("description", r.get("final_output", ""))

    elif task == "table_recommender":
        raw = r.get("components", {}).get("recommended_tables", [])
        r["predicted_tables"] = _parse_orca_recommended_tables(raw)
        if ground_truth_map is not None:
            r["ground_truth"] = ground_truth_map.get(r.get("question"), [])

    return r


# ----- EVALUATION FUNCTIONS -----

def grade_table_description(description: str, grader_llm) -> dict:
    """Grade a table description on a 0–5 rubric. Returns {score, reason}."""
    if not description:
        return {"score": 0, "reason": "No description provided"}

    chain = table_description_grading_prompt | grader_llm
    try:
        result = chain.invoke({"table_description": description})
        response = result.content.strip()
        if response.startswith("```"):
            lines = response.split("\n")
            response = "\n".join(
                lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
            )
        return json.loads(response)
    except json.JSONDecodeError:
        return {
            "score": 0,
            "reason": f"Failed to parse grader response: {result.content[:100]}",
        }
    except Exception as e:
        return {"score": 0, "reason": f"Grading error: {e}"}


def eval_table_recommender(predicted: list, ground_truth: list) -> dict:
    """Compute precision, recall, and F1 for table recommendation."""
    pred_set = set(predicted) if predicted else set()
    gold_set = set(ground_truth) if ground_truth else set()

    if not pred_set and not gold_set:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    tp = len(pred_set & gold_set)
    precision = tp / len(pred_set) if pred_set else 0.0
    recall = tp / len(gold_set) if gold_set else 0.0
    f1 = (
        (2 * precision * recall / (precision + recall))
        if (precision + recall) > 0
        else 0.0
    )
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def eval_text2sql(pred_sql: str, gold_sql: str, db_id: str) -> dict:
    """Execute both SQL queries and compare result sets (execution accuracy)."""
    if not pred_sql:
        return {"match": False, "error": "No predicted SQL"}

    db = Database()
    try:
        pred_rows, _ = db.run_query(pred_sql, db_id)
        gold_rows, _ = db.run_query(gold_sql, db_id)

        pred_sorted = sorted(
            [tuple(str(v) if v is not None else "NULL" for v in row) for row in pred_rows]
        )
        gold_sorted = sorted(
            [tuple(str(v) if v is not None else "NULL" for v in row) for row in gold_rows]
        )
        return {"match": pred_sorted == gold_sorted, "error": None}
    except Exception as e:
        return {"match": False, "error": str(e)}


# ----- MAIN EVALUATION RUNNER -----

def run_evaluation(
    results_path: str,
    task: str,
    db_id: str = None,
    grader_model: str = "gpt-4o-mini",
    grader_provider: str = "openai",
    source: str = "baseline",
    questions_path: str = None,
) -> dict:
    """
    Load a results JSON, evaluate each entry, and write summary + per-entry eval.

    source         : "baseline" (run_baseline.py output) or "orca" (run_experiments.py output)
    questions_path : path to table_recommendation.json — required for ORCA table_recommender
                     so that ground-truth answers can be loaded.
    Returns the summary dict.
    """
    results_path = Path(results_path)
    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    # Build ground-truth map for ORCA table_recommender
    ground_truth_map = None
    if source == "orca" and task == "table_recommender":
        if not questions_path:
            raise ValueError(
                "--questions-path is required when evaluating ORCA table_recommender results "
                "(needed to load ground-truth table lists)"
            )
        with open(questions_path, "r", encoding="utf-8") as f:
            ground_truth_map = {q["question"]: q["answer"] for q in json.load(f)}

    # Normalise ORCA results to the flat format the eval functions expect
    if source == "orca":
        results = [normalize_orca_result(r, task, ground_truth_map) for r in results]

    evaluated = []

    if task == "table_explorer":
        grader_llm = get_llm(model=grader_model, temperature=0.0, provider=grader_provider)
        scores = []
        for r in results:
            eval_result = grade_table_description(r.get("description"), grader_llm)
            r["eval"] = eval_result
            scores.append(eval_result.get("score", 0))
            evaluated.append(r)
            print(f"  {r['table_name']}: score={eval_result.get('score')}  — {eval_result.get('reason', '')[:60]}")

        n = len(scores)
        summary = {
            "task": task,
            "n": n,
            "mean_score": round(sum(scores) / n, 4) if n else 0,
            "score_distribution": {str(i): scores.count(i) for i in range(6)},
        }

    elif task == "table_recommender":
        prec_list, rec_list, f1_list = [], [], []
        for r in results:
            metrics = eval_table_recommender(
                r.get("predicted_tables", []), r.get("ground_truth", [])
            )
            r["eval"] = metrics
            prec_list.append(metrics["precision"])
            rec_list.append(metrics["recall"])
            f1_list.append(metrics["f1"])
            evaluated.append(r)
            print(
                f"  P={metrics['precision']:.2f} R={metrics['recall']:.2f} "
                f"F1={metrics['f1']:.2f}  Q: {r['question'][:50]}..."
            )

        n = len(f1_list)
        summary = {
            "task": task,
            "n": n,
            "mean_precision": round(sum(prec_list) / n, 4) if n else 0,
            "mean_recall": round(sum(rec_list) / n, 4) if n else 0,
            "mean_f1": round(sum(f1_list) / n, 4) if n else 0,
        }

    elif task == "text2sql":
        matches = []
        by_difficulty: dict = {}
        for r in results:
            effective_db_id = db_id or r.get("db_id")
            eval_result = eval_text2sql(
                r.get("generated_sql"), r["ground_truth_sql"], effective_db_id
            )
            r["eval"] = eval_result
            matches.append(eval_result["match"])
            diff = r.get("difficulty", "unknown")
            by_difficulty.setdefault(diff, []).append(eval_result["match"])
            evaluated.append(r)
            status = "✓" if eval_result["match"] else "✗"
            err = f"  ({eval_result['error']})" if eval_result.get("error") else ""
            print(f"  [{status}] Q{r['question_id']}: {r['question'][:50]}...{err}")

        n = len(matches)
        summary = {
            "task": task,
            "n": n,
            "execution_accuracy": round(sum(matches) / n, 4) if n else 0,
            "by_difficulty": {
                diff: round(sum(ms) / len(ms), 4) if ms else 0
                for diff, ms in by_difficulty.items()
            },
        }

    else:
        raise ValueError(
            f"Unknown task: '{task}'. Choose from: table_explorer, table_recommender, text2sql"
        )

    eval_path = results_path.parent / (results_path.stem + "_eval.json")
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": evaluated}, f, indent=2, ensure_ascii=False)

    print(f"\nSummary:\n{json.dumps(summary, indent=2)}")
    print(f"Saved to: {eval_path}")
    return summary


# ----- MAIN -----

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate data exploration experiment results (baseline or ORCA)"
    )
    parser.add_argument("results_path", type=str, help="Path to the results JSON file")
    parser.add_argument(
        "-t", "--task", type=str, required=True,
        choices=["table_explorer", "table_recommender", "text2sql"],
    )
    parser.add_argument(
        "--source", type=str, default="baseline", choices=["baseline", "orca"],
        help="Result format: 'baseline' (run_baseline.py) or 'orca' (run_experiments.py)"
    )
    parser.add_argument(
        "--db-id", type=str, default=None,
        help="Override db_id for text2sql execution (default: uses value in results file)"
    )
    parser.add_argument(
        "--grader-model", type=str, default="gpt-4o-mini",
        help="Model for table_explorer grading"
    )
    parser.add_argument(
        "--grader-provider", type=str, default="openai",
        choices=["openai", "google", "ollama"],
    )
    parser.add_argument(
        "--questions-path", type=str, default=None,
        help="Path to table_recommendation.json — required for --source orca --task table_recommender"
    )
    args = parser.parse_args()

    run_evaluation(
        results_path=args.results_path,
        task=args.task,
        db_id=args.db_id,
        grader_model=args.grader_model,
        grader_provider=args.grader_provider,
        source=args.source,
        questions_path=args.questions_path,
    )


if __name__ == "__main__":
    main()
