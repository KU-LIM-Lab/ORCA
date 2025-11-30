import json
from tqdm import tqdm
import pandas as pd
import numpy as np

from experiments.causal_analysis.causal_pre_information import DEFAULT_EXPRESSION_DICT, BASE_SQL_QUERY
from agents.causal_analysis.graph import generate_causal_analysis_graph
from agents.text2sql_generator.nodes.selector import get_schema_summary
from utils.llm import get_llm
from utils.load_causal_graph import load_causal_graph
from utils.database import Database

import os
from datetime import datetime
import warnings

# warnings.filterwarnings("ignore", category=FutureWarning, module="dowhy")

# Load LLM 
llm = get_llm(model="gpt-4o-mini", temperature=0.7, provider="openai")

# Prepare input
causal_graph = load_causal_graph("experiments/causal_analysis/causal_graph_full.json")
app = generate_causal_analysis_graph(llm=llm)

with open("all_db_ddls.json") as f:
    all_ddl = json.load(f)
ddl_str = json.dumps(all_ddl["daa"], indent=2)

# Queries
input_path = "experiments/questions/daa/causal_analysis.json"

with open(input_path, "r") as f:
    queries = json.load(f)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
result_path = f"experiments/results/baseline_eval_results_{timestamp}.jsonl"
summary_path = f"experiments/results/baseline_eval_summary_{timestamp}.json"
os.makedirs(os.path.dirname(result_path), exist_ok=True)
open(result_path, "w").close()

# Retrieve data sample
database = Database()
df = database.run_query(sql=BASE_SQL_QUERY, db_id="daa")
df = pd.DataFrame(df[0], columns=df[1])
sample_data = df.head(10).to_dict(orient="records")

table_names = database.list_tables(db_id="daa")
schema_info, fk_info, schema_tables = get_schema_summary("daa", table_names)
table_schema_str = schema_info + "\n\n" + fk_info

N = 10
summary_list = []

baseline_mae_list, baseline_mse_list, baseline_maxae_list = [], [], []
agent_mae_list, agent_mse_list, agent_maxae_list = [], [], []

for run_idx in range(N):
    print(f"\n🔁 Run {run_idx + 1}/{N}")
    total, success_count, baseline_within_ci_count, agent_within_ci_count = 0, 0, 0, 0
    
    baseline_abs_errors, agent_abs_errors = [], []

    for q in tqdm(queries, desc=f"Evaluating causal questions (Run {run_idx + 1})"):
        total += 1
        question = q["question"]
        try:
            # basic information
            treatment = q["treatment"]
            outcome = q["outcome"]
            confounders = q.get("confounders", [])
            mediators = q.get("mediators", [])
            instrumental_variables = q.get("instrumental_variables", [])
            treatment_expr = DEFAULT_EXPRESSION_DICT.get(treatment, treatment)
            outcome_expr = DEFAULT_EXPRESSION_DICT.get(outcome, outcome)
            confounder_exprs = [DEFAULT_EXPRESSION_DICT.get(c, c) for c in confounders]

            # === 1. GPT-4o-mini baseline ===
            graph_text = "Causal Graph (source → target):\n"
            graph_text += "\n".join([f"- {src} → {dst}" for src, dst in causal_graph["edges"]])

            prompt = f"""
You are a data analyst. Based on the question below, estimate the causal effect using the provided data, schema, and causal graph.

Question: {question}

Causal Variables:
- Treatment: {treatment}
- Outcome: {outcome}
- Confounders: {confounders}

Causal Graph (source → target):
{graph_text}

DDL Schema (PostgreSQL format):
{ddl_str}

Sample Data (Top 10 rows):
{json.dumps(sample_data, indent=2, default=str)}

Please output only a JSON object with the following keys:
- "estimated_ate": a float rounded to 3 decimal places
- "confidence_interval": a list of two floats, both rounded to 3 decimal places

Do not include any explanation or markdown formatting.
Only return the raw JSON (no triple backticks).
"""
            baseline_response = llm.invoke(input = prompt)
            content = baseline_response.content
            baseline_parsed = json.loads(content)
            baseline_ate = baseline_parsed["estimated_ate"]
            baseline_ci = baseline_parsed["confidence_interval"]

            # === 2. Agent run ===
            state_input = {
                "input": question,
                "db_id": "daa",
                "variable_info": {
                    "treatment": treatment,
                    "treatment_expression": treatment_expr,
                    "outcome": outcome,
                    "outcome_expression": outcome_expr,
                    "confounders": confounders,
                    "confounder_expressions": confounder_exprs,
                    "mediators": mediators,
                    "instrumental_variables": instrumental_variables
                },
                "causal_graph": causal_graph,
                "expression_dict": DEFAULT_EXPRESSION_DICT,
                "schema_info": table_schema_str,
                "sql_query": BASE_SQL_QUERY # if you want to run agentic version, delete this line
            }

            result = app.invoke(state_input)
            agent_ate = result.get("causal_effect_value") or result.get("causal_effect_ate")
            agent_ci = result.get("confidence_interval")
            agent_p_value = result.get("causal_effect_p_value")

            # === 평가 ===
            ground_truth_ate = q.get("ground_truth_ate")
            ground_truth_ci = q.get("confidence_interval")
            if isinstance(ground_truth_ci[0], list):
                ground_truth_ci = ground_truth_ci[0]

            baseline_within_gt_ci = ground_truth_ci[0] <= baseline_ate <= ground_truth_ci[1]
            agent_within_gt_ci = agent_ate is not None and ground_truth_ci[0] <= agent_ate <= ground_truth_ci[1]

            success_count += 1
            if agent_within_gt_ci:
                agent_within_ci_count += 1
            if baseline_within_gt_ci:
                baseline_within_ci_count += 1

            baseline_abs_errors.append(abs(baseline_ate - ground_truth_ate))
            if agent_ate is not None:
                agent_abs_errors.append(abs(agent_ate - ground_truth_ate))

            result_dict = {
                "question": question,
                "success": True,
                "ground_truth_ate": ground_truth_ate,
                "ground_truth_ci": ground_truth_ci,

                "baseline_ate": baseline_ate,
                "baseline_ci": baseline_ci,
                "baseline_within_gt_ci": baseline_within_gt_ci,
                "baseline_absolute_error": abs(baseline_ate - ground_truth_ate),

                "agent_ate": agent_ate,
                "agent_ci": agent_ci,
                "agent_within_gt_ci": agent_within_gt_ci,
                "agent_absolute_error": abs(agent_ate - ground_truth_ate) if agent_ate is not None else None,
                "agent_p_value": agent_p_value,
                "agent_method": result.get("estimation_method"),
                "ground_truth_method": q.get("estimation_method"),
                "run_idx": run_idx + 1
            }

        except Exception as e:
            result_dict = {
                "question": question,
                "success": False,
                "error_message": str(e),
                "run_idx": run_idx + 1
            }

        with open(result_path, "a") as f:
            f.write(json.dumps(result_dict) + "\n")


    # Summary stats
    summary = {
        "total": total,
        "success_count": success_count,
        "baseline_ci_coverage_count": baseline_within_ci_count,
        "agent_ci_coverage_count": agent_within_ci_count,
        "baseline_success_rate": round(baseline_within_ci_count / total * 100, 2),
        "agent_success_rate": round(agent_within_ci_count / success_count * 100, 2) if success_count > 0 else 0.0,
    }  
    summary_list.append(summary)

    baseline_mae_list.append(np.mean(baseline_abs_errors))
    baseline_mse_list.append(np.mean(np.square(baseline_abs_errors)))
    baseline_maxae_list.append(np.max(baseline_abs_errors))
    agent_mae_list.append(np.mean(agent_abs_errors))
    agent_mse_list.append(np.mean(np.square(agent_abs_errors)))
    agent_maxae_list.append(np.max(agent_abs_errors))

# Compute final CI metrics
def compute_mean_ci(values):
    arr = np.array(values)
    mean = np.mean(arr)
    ci = 1.96 * np.std(arr, ddof=1) / np.sqrt(N)
    return mean, ci

metrics = ["baseline_success_rate", "agent_success_rate"]
final_summary = {}
print("\n Final Evaluation Summary (Mean ± 95% CI):")
for m in metrics:
    values = [s[m] for s in summary_list]
    mean, ci = compute_mean_ci(values)
    final_summary[m] = f"{mean} ± {ci}"
    print(f"{m}: {final_summary[m]}")

def compute_ci_label(name, values):
    mean, ci = compute_mean_ci(values)
    return f"{name}: {mean:.6e} ± {ci:.6e}"

final_summary.update({
    "baseline_mae": f"{compute_mean_ci(baseline_mae_list)[0]:.6e} ± {compute_mean_ci(baseline_mae_list)[1]:.6e}",
    "baseline_mse": f"{compute_mean_ci(baseline_mse_list)[0]:.6e} ± {compute_mean_ci(baseline_mse_list)[1]:.6e}",
    "baseline_maxae": f"{compute_mean_ci(baseline_maxae_list)[0]:.6e} ± {compute_mean_ci(baseline_maxae_list)[1]:.6e}",
    "agent_mae": f"{compute_mean_ci(agent_mae_list)[0]:.6e} ± {compute_mean_ci(agent_mae_list)[1]:.6e}",
    "agent_mse": f"{compute_mean_ci(agent_mse_list)[0]:.6e} ± {compute_mean_ci(agent_mse_list)[1]:.6e}",
    "agent_maxae": f"{compute_mean_ci(agent_maxae_list)[0]:.6e} ± {compute_mean_ci(agent_maxae_list)[1]:.6e}",
})

with open(summary_path, "w") as f:
    json.dump(final_summary, f, indent=2)

print(f"\n✅ Evaluation complete with {N} runs.")
print(f"Summary saved to: {summary_path}")