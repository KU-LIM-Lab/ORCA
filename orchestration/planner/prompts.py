# orchestration/planner/prompts.py
"""
Prompts for the planner agent to analyze user queries and create execution plans
"""

PLANNER_SYSTEM_PROMPT = """
You are an expert planning agent for the ORCA (ORchestrated Causal Analysis) system. 
Your role is to analyze user queries and create detailed execution plans for causal analysis workflows.

Available Analysis Types:
1. Data Exploration: Understanding data structure, quality, and basic statistics
2. Causal Discovery: Finding causal relationships and generating causal graphs
3. Causal Inference: Estimating causal effects and performing quantitative analysis
4. Full Analysis: Complete end-to-end causal analysis pipeline

Available Agents:
- data_explorer: Database exploration, schema analysis, data preprocessing
- causal_discovery: Causal graph discovery, algorithm selection, data diagnosis
- causal_inference: Causal effect estimation, sensitivity analysis, robustness checks

Your task is to:
1. Analyze the user's query to understand their intent
2. Determine which analysis steps are needed
3. Create a detailed execution plan with proper dependencies
4. Consider potential error scenarios and recovery strategies
5. Optimize the plan for efficiency and accuracy

Always consider the user's specific needs and provide a plan that addresses their exact requirements.
"""

QUERY_ANALYSIS_PROMPT = """
Analyze the following user query and determine the required analysis type and steps:

User Query: "{query}"

Context: {context}

Please provide:
1. Analysis Type: [data_exploration|causal_discovery|causal_inference|full_analysis]
2. Required Agents: List of agents needed
3. Key Requirements: Specific requirements from the query
4. Potential Challenges: Possible issues or limitations
5. Suggested Approach: High-level approach for the analysis

Format your response as JSON.
"""

PLAN_CREATION_PROMPT = """
Create a detailed execution plan based on the following analysis:

Query: "{query}"
Analysis Type: {analysis_type}
Required Agents: {required_agents}
Context: {context}

Create a step-by-step execution plan with:
1. Step ID: Unique identifier
2. Agent: Which agent to use
3. Action: Specific action to perform
4. Description: Human-readable description
5. Dependencies: Previous steps that must complete first
6. Expected Output: What this step should produce
7. Timeout: Maximum time allowed (in seconds)
8. Error Handling: How to handle potential errors

Format as JSON array of step objects.
"""

ERROR_RECOVERY_PROMPT = """
The execution has encountered an error. Create a recovery plan:

Error Details:
- Step: {failed_step}
- Error Type: {error_type}
- Error Message: {error_message}
- Completed Steps: {completed_steps}
- Current State: {current_state}

Create a recovery plan that:
1. Addresses the specific error
2. Minimizes data loss
3. Provides alternative approaches
4. Maintains analysis integrity

Format as JSON array of recovery step objects.
"""

USER_FEEDBACK_PROMPT = """
The user has provided feedback during execution. Adjust the plan accordingly:

Original Query: "{original_query}"
Current Plan: {current_plan}
User Feedback: "{user_feedback}"
Completed Steps: {completed_steps}
Current State: {current_state}

Based on the feedback, create an updated plan that:
1. Incorporates the user's feedback
2. Maintains consistency with previous steps
3. Addresses any new requirements
4. Optimizes for the user's preferences

Format as JSON array of updated step objects.
"""

# Query classification patterns
QUERY_PATTERNS = {
    "data_exploration": [
        "데이터 특성", "데이터 분포", "테이블 구조", "스키마 분석",
        "데이터 탐색", "데이터 요약", "데이터 설명", "어떤 데이터",
        "데이터 품질", "결측치", "이상치", "기본 통계"
    ],
    "causal_discovery": [
        "인과 관계", "원인과 결과", "영향 요인", "왜 발생",
        "어떤 원인", "요인 분석", "causal", "인과 발견",
        "그래프 생성", "관계 분석", "연관성"
    ],
    "causal_inference": [
        "인과 효과", "효과 측정", "정량 분석", "만약 한다면",
        "가정 분석", "시나리오", "inference", "추론",
        "추정", "정량적", "효과 크기"
    ],
    "full_analysis": [
        "전체 분석", "종합 분석", "완전 분석", "데이터부터 결과까지",
        "end to end", "전체 프로세스", "완전한 분석", "종합적"
    ]
}

# Agent capability mapping
AGENT_CAPABILITIES = {
    "data_explorer": {
        "primary": ["데이터 탐색", "스키마 분석", "데이터 전처리"],
        "secondary": ["SQL 생성", "데이터 품질 검사", "통계 요약"],
        "outputs": ["data_explored", "schema_info", "preprocessed_data"]
    },
    "causal_discovery": {
        "primary": ["인과 그래프 발견", "알고리즘 선택", "데이터 가정 검증"],
        "secondary": ["그래프 평가", "알고리즘 비교", "가정 검증"],
        "outputs": ["causal_graph", "algorithm_scores", "assumption_validation"]
    },
    "causal_inference": {
        "primary": ["인과 효과 추정", "정량 분석", "민감도 분석"],
        "secondary": ["로버스트니스 검사", "신뢰구간", "효과 크기"],
        "outputs": ["causal_estimates", "confidence_intervals", "sensitivity_results"]
    }
}

# Timeout configurations (in seconds)
TIMEOUT_CONFIGS = {
    "data_exploration": 300,  # 5 minutes
    "causal_discovery": 600,  # 10 minutes
    "causal_inference": 900,  # 15 minutes
    "report_generation": 300, # 5 minutes
    "error_recovery": 180     # 3 minutes
}

# Error handling strategies
ERROR_STRATEGIES = {
    "data_access_error": {
        "recovery": "retry_data_access",
        "fallback": "use_alternative_data_source",
        "timeout": 180
    },
    "causal_discovery_failed": {
        "recovery": "try_alternative_algorithms",
        "fallback": "use_manual_graph_construction",
        "timeout": 600
    },
    "causal_inference_failed": {
        "recovery": "adjust_estimation_method",
        "fallback": "use_simpler_estimation",
        "timeout": 300
    },
    "timeout_error": {
        "recovery": "increase_timeout_and_retry",
        "fallback": "use_faster_algorithm",
        "timeout": 300
    }
}
