# nodes/generate_answer.py

from typing import Dict
from langchain_core.runnables import RunnableLambda
from langchain_core.language_models.chat_models import BaseChatModel

from utils.llm import call_llm
from prompts.causal_analysis_prompts import generate_answer_prompt, generate_answer_parser


def build_generate_answer_node(llm: BaseChatModel) -> RunnableLambda:
    """
    Uses an LLM to generate an interpretable explanation of the causal inference results.
    """

    def invoke(state: Dict) -> Dict:
        strategy = state["strategy"]
        parsed_query = state.get("parsed_query") or {}
        estimate = state.get("causal_estimate")
        refutation_result = state.get("refutation_result")
        label_maps = state.get("label_maps")

        if not strategy or not estimate:
            raise ValueError("Missing strategy or causal_estimate")

        llm_input = {
            "task": strategy.task,
            "estimation_method": strategy.estimator,
            "treatment": parsed_query["treatment"],
            "treatment_expression_description": parsed_query.get("treatment_expression_description", ""),
            "outcome": parsed_query["outcome"],
            "outcome_expression_description": parsed_query.get("outcome_expression_description", ""),
            "confounders": parsed_query.get("confounders", []),
            "mediators": parsed_query.get("mediators", []),
            "instrumental_variables": parsed_query.get("instrumental_variables", []),
            "refutation_result": refutation_result if refutation_result else None,
            "label_maps": label_maps if label_maps else None,
            "causal_effect_value": state["causal_effect_ate"],
            "causal_effect_ci": state["causal_effect_ci"],
        }

        result = call_llm(
            prompt=generate_answer_prompt,
            parser=generate_answer_parser,
            variables=llm_input,
            llm=llm
        )

        state["final_answer"] = result
        state["final_answer"] = result.explanation if hasattr(result, "explanation") else str(result)
        return state

    return RunnableLambda(invoke)