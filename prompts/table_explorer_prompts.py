from typing import List, Union
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate


# column description prompt and parser 
class ColumnDescription(BaseModel):
    column_name: str
    data_type: str
    nullable: str  # YES or NO
    nulls: Union[int, str]  # Mostly int but sometimes can be unknown
    notes: List[str]

class TableAnalysis(BaseModel):
    table_description: str
    columns: List[ColumnDescription]
    analysis_considerations: str

column_description_parser = PydanticOutputParser(pydantic_object=TableAnalysis)

system_template = """You are a professional data analyst specialized in database documentation and exploratory data analysis (EDA).

You will be given a table name and a summary of its columns.
Your task is to provide a structured JSON analysis consisting of the following fields:

1. table_description: (string, 1–2 sentences)
    - Briefly describe the type of data stored, who generates it, and its primary analytical uses.

2. columns: (list of objects)
    For each column, provide the following:
    - column_name: str
    - data_type: str
    - nullable: "YES" or "NO"
    - nulls: int or "unknown"
    - notes: List[str] — analytical observations such as:
        - High null ratio (e.g., "Nulls: 340/1000 (34%)")
        - Distinct count and duplicates (e.g., "Distinct values: 983/1000 → some duplicates")
        - Low variance or uniformity (e.g., "All values are 'true'")
        - Range coverage (e.g., "Date range from 1970-01-10 to 2005-12-01")
        - Median, min/max (e.g., "Median birth date is 1988-05-18")
        - Potential outliers or suspicious values, other range issues
        - Low variance, skewness, duplicates; Uniform or skewed distribution based on median, mean, min/max for numerical columns, and top value counts for categorical columns.
    
      Notes must:
      - Provide meaningful insights from a data analysis perspective (e.g., likelihood of being an identifier, presence of outliers, skewed distributions, repeated values, narrow/wide range issues).
      - When possible, **include specific statistics** (e.g., counts, percentages, ranges, min/max/mean/median).
      - Avoid vague language like “high” or “low” without numeric support.
      - Be practical advice for downstream analysis, not superficial descriptions.
      - If multiple notes exist, number them clearly as "1.", "2.", "3.", etc.

3. analysis_considerations: (string)
    - Summarize any critical risks, biases, or opportunities associated with this table.
    - Mention how the quality of this table may affect downstream analysis.
    - Mention any prepocessing or transformations that may be needed.

4. recommended_analyses: (list of objects)
    Each recommendation should include:
    - name: str
    - objective: str
    - data_requirements: List[str] (table/column names) (e.g., "users.point_balance", "orders.total_price")
    - method: str (e.g., cohort analysis, regression, survival analysis)
    - assumptions: str
    - expected_insights: str

Output must conform exactly to the JSON structure defined above.
{format_instructions}
"""

human_template = """
[Table Name]: {table_name}

[Column Summary Information]:
{column_summary}
"""

column_description_prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template(human_template)
    ],
    input_variables=["table_name", "column_summary"],
    partial_variables={"format_instructions": column_description_parser.get_format_instructions()}
)


# Usecase recommendation prompt and parser
class Usecase(BaseModel):
    analysis_topic: str = Field(..., alias="Analysis Topic")
    suggested_methodology: str = Field(..., alias="Suggested Methodology")
    expected_insights: str = Field(..., alias="Expected Insights")

class RecommendedUsecases(BaseModel):
    usecases: list[Usecase] = Field(..., alias="Usecases")
    class Config:
        populate_by_name = True  # Allow values to be received via alias as well

usecase_parser = PydanticOutputParser(pydantic_object=RecommendedUsecases)

system_template = """
You are a business data analyst.

Your job is to generate up to 3 specific, concrete analysis Usecases based on:
- A table description
- Related tables information

For each idea, include:
- Analysis Topic: (what specific relationship or pattern you propose to study)
- Suggested Methodology: (what kind of statistical analysis, aggregation, segmentation, or modeling would be appropriate)
- Expected Insights: (what kind of business value or decision-making this analysis could support)

Respond in a numbered list. Avoid vague suggestions.
{format_instructions}
"""

human_template = """
[Table Description]
{table_description}

[Related Tables]
{related_tables}
"""

usecase_prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template(human_template)
    ],
    input_variables=["table_description", "related_tables"],
    partial_variables={"format_instructions": usecase_parser.get_format_instructions()}
)
