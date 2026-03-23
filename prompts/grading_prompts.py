# ORCA/prompts/grading_prompts.py

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate


# Table Explorer Grading Prompt (5-point scale)
table_description_grading_system = """You are a data expert tasked with evaluating the analytical quality of a table description.

You will be given a `table_description` and must assign a score from 0 to 5 based on how well the description:
- Captures outliers
- Analyzes missing values
- Describes distributional characteristics
- Suggests appropriate analysis directions or methods

Go beyond surface-level summaries. Evaluate whether the author shows an understanding of how the data should be interpreted and analyzed, not just what it looks like.

Scoring Rubric (0–5 points):

0 points:
- Purely structural description with no mention of missing values, outliers, or data distributions.
- Only lists column types, basic metadata, or high-level table purpose.

1 point:
- Mentions some missing values or simple statistics (e.g., number of distinct values).
- Lacks any interpretation or analytical context.

2 points:
- Includes null ratios or range for some columns.
- May suggest that missing data or certain value ranges exist, but no specific interpretation or analysis method is provided.

3 points:
- Consistently describes missing values and value distributions across most columns.
- Identifies possible outliers or highly skewed data.
- Mentions basic statistics like mean, median, or top values.
- Provides minor analytical hints but no concrete method suggestions.

4 points:
- Well-rounded analysis of missingness, outliers, and distributions across important columns.
- Offers useful guidance on potential data preprocessing (e.g., log transformation, filtering).
- Considers how data quality or structure affects downstream analysis.

5 points:
- Comprehensive analytical insights, including nulls, outliers, and value distributions.
- Recommends appropriate analysis methods (e.g., regression, classification, time series).
- Discusses assumptions, interpretation directions, or modeling considerations.
- Connects column-level insights to potential analytical goals or questions.

Respond with ONLY a JSON object: {{"score": <int>, "reason": "<brief justification>"}}
"""

table_description_grading_human = """
[Table Description to Evaluate]:
{table_description}
"""

table_description_grading_prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(table_description_grading_system),
        HumanMessagePromptTemplate.from_template(table_description_grading_human),
    ],
    input_variables=["table_description"],
)