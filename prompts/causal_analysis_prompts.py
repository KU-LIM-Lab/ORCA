from typing import List, Optional
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


### 1.1. Treatment/Outcome Identification Prompt
class TreatmentOutcomeIdentification(BaseModel):
    treatment: str = Field(..., description="The treatment variable name (variable being manipulated)")
    outcome: str = Field(..., description="The outcome variable name (variable being affected)")
    reasoning: str = Field(..., description="Brief explanation of why these variables were selected")

identify_treatment_outcome_parser = PydanticOutputParser[TreatmentOutcomeIdentification](pydantic_object=TreatmentOutcomeIdentification)

identify_treatment_outcome_prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are an expert in causal inference. Your task is to identify the treatment and outcome variables from a user's causal question and available data.

Given:
- A natural language causal question
- A sample of the preprocessed DataFrame
- Table schema information

Identify:
- **Treatment (T)**: The variable that is being manipulated or intervened upon (the "cause")
- **Outcome (Y)**: The variable that is being measured or affected (the "effect")

Rules:
- Treatment should be a variable that can be manipulated or assigned (e.g., "received_treatment", "discount_applied", "campaign_exposed")
- Outcome should be a variable that represents the result or effect (e.g., "purchase_amount", "conversion_rate", "customer_satisfaction")
- Use variable names that exist in the DataFrame columns
- Provide clear reasoning for your selection

{format_instructions}
"""),
    ("human", """
User Question:
{question}

DataFrame Sample:
{df_sample}

Table Schema:
{tables}

Available Columns:
{columns}
""")
]).partial(format_instructions=identify_treatment_outcome_parser.get_format_instructions())

### 2. SQL Generation Prompt
sql_generation_prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are a senior data analyst generating a PostgreSQL query for causal analysis. Decompose the task into subtasks to generate a final SQL query.
     
You will be given:
- A list of variable names that appear in the causal graph with a mapping to its SQL expression. Every variable **must** be selected in the final query **only** using its corresponding SQL expression. Do not use other expressions, even if they are semantically equivalent.
- Table Schema Information.

## Instructions:
Decompose the task into:
1.	For each variable, check if the expression can be computed directly from the available tables by a simple join (and, if necessary, GROUP BY or aggregate). If so, compute it directly in the final_table CTE and do not create a separate CTE for this variable.
2.  If a variable cannot be computed directly from the available tables and requires a non-trivial subquery, repeated logic, or pre-aggregation that cannot be handled directly in final_table create a CTE for that variable. If multiple variables share the same SQL expression, they can be selected together in the same CTE/subquery. 
3.	In final_table, join all necessary tables and CTEs and select all variables. The order in which you join tables in the final_table should always respect the foreign key (FK) relationships specified in the schema.
4.	The final query must always be SELECT * FROM final_table;.

## Rules:
- Do not include explanatory text
- NEVER assume any relationships not explicitly stated as foreign keys. When joining CTEs or tables, use only the foreign key relationships defined in the schema. Do not calculate a new varaible to join tables.
- Do not alias tables (e.g., avoid `FROM users u`)
- Do not include GROUP BY or ORDER BY unless needed
- NEVER write expressions like `(SELECT ... WHERE ... = outer_column LIMIT 1)`, as they will result in 'correlated subquery' errors in PostgreSQL.
- Always output the final SQL query as a code block with ```sql ```.

## Example1:
Variables:
- cart_item_id
- campaign_exposed
- cart_total_value

Variable to SQL expression mapping:
- cart_item_id: cart_items.cart_item_id
- campaign_exposed: CASE WHEN COUNT(campaign_exposure.exposure_id) > 0 THEN TRUE ELSE FALSE END
- cart_total_value: SUM(cart_items.price * cart_items.quantity)

Schema:
TABLE: carts
  COLUMNS: cart_id (uuid), customer_id (uuid), created_at (date)
TABLE: cart_items
  COLUMNS: cart_item_id (uuid), cart_id (uuid), product_id (uuid), price (numeric), quantity (integer)
TABLE: campaign_exposure
  COLUMNS: exposure_id (uuid), cart_id (uuid), campaign_id (uuid)
TABLE: campaigns
  COLUMNS: campaign_id (uuid), name (text), start_date (date), end_date (date)

FOREIGN KEYS:
  carts.cart_id -> cart_items.cart_id
  carts.cart_id -> campaign_exposure.cart_id
  campaign_exposure.campaign_id -> campaigns.campaign_id

Output:
```sql
WITH
campaign_exposed_per_cart AS (
    SELECT
        cart_id,
        CASE WHEN COUNT(exposure_id) > 0 THEN TRUE ELSE FALSE END AS campaign_exposed
    FROM campaign_exposure
    GROUP BY cart_id
),
final_table AS (
    SELECT
        cart_items.cart_item_id AS cart_item_id,
        SUM(cart_items.price * cart_items.quantity) AS cart_total_value
        campaign_exposed_per_cart.campaign_exposed AS campaign_exposed
    FROM cart_items 
    LEFT JOIN campaign_exposed_per_cart ON cart_items.cart_id = campaign_exposed_per_cart.cart_id
)
SELECT * FROM final_table;
```

## Example2:
Variables:
- student_id
- average_score
- teacher_name

Variable to SQL expression mapping:
- student_id: students.student_id
- average_score: AVG(scores.score)
- teacher_name: teachers.name

Schema:
TABLE: students
  COLUMNS: student_id (uuid), name (text), class_id (uuid)
TABLE: classes
  COLUMNS: class_id (uuid), teacher_id (uuid)
TABLE: teachers
  COLUMNS: teacher_id (uuid), name (text)
TABLE: scores
  COLUMNS: score_id (uuid), student_id (uuid), score (numeric)

FOREIGN KEYS:
  students.class_id -> classes.class_id
  classes.teacher_id -> teachers.teacher_id
  scores.student_id -> students.student_id

Output:
```sql
WITH
average_score_table AS (
    SELECT
        students.student_id,
        AVG(scores.score) AS average_score
    FROM students
    LEFT JOIN scores ON students.student_id = scores.student_id
    GROUP BY students.student_id
),
final_table AS (
    SELECT
        students.student_id AS student_id,
        average_score.average_score AS average_score,
        teachers.name AS teacher_name
    FROM students 
    LEFT JOIN average_score_table ON students.student_id = average_score_table.student_id
    LEFT JOIN classes ON students.class_id = classes.class_id -- need to join classes before joining teachers to get teacher_id (FK chain)
    LEFT JOIN teachers ON classes.teacher_id = teachers.teacher_id 
)
SELECT * FROM final_table;
```
"""),
    ("human", """
Variables:
{selected_variables}

Variable to SQL expression mapping:
{variable_expressions}

Schema:
{table_schemas}
""")
])

### 3. review SQL Prompt
sql_review_prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are a senior data analyst reviewing a PostgreSQL query for causal analysis.
     
You will be given:
1. A SQL query that selects data for causal analysis
2. A list of variable names that must be selected in the query and their corresponding SQL expressions.
3. Table schemas

## instructions:   
You task is to check if the SQL query meets the following requirements:
    1. The query must select all variables listed in 【Variables】. Also, you must check that each variable is selected using **exactly** the SQL expression provided in 【Variable SQL Expressions】. However, it is allowed for the required SQL expression to appear in a CTE (Common Table Expression) or subquery, as long as the final output selects the variable using the correct alias that refers to the CTE/subquery result.
    - If the same variable name appears multiple times in the SELECT clause, keep only the one that uses the exact required SQL expression, and remove all others. Update the JOIN conditions if needed.
    2. The query should only use JOINs based on explicit foreign key (FK) relationships as defined in the table schemas. If any JOINs are not based on an FK relationship, modify the query to use the correct FK-based JOIN instead.
     - When fixing Joins, consider the order of joins in the FROM clause. The order should respect the foreign key (FK) relationships specified in the schema.
    3. If sql needs to be fixed, output the revised SQL query in a code block with ```sql ```.

## Rules:
- When revising the query, you must preserve the original overall structure as much as possible (including CTEs, JOINs, and the general flow). Only add or modify specific parts as needed to satisfy the requirements. Do not rewrite the entire query from scratch.
- If the same variable name appears multiple times in the SELECT clause, keep only the one that uses the exact required SQL expression, and remove all others. Update the JOIN conditions if needed.
- Don't include variables other than those listed in 【Variables】. But you must ensure that all variables in 【Variables】 are selected in the final query using the SQL Expressions.
- Never select the same variable multiple times in the final query.  
     
## Example1:
【Variables】
- cart_item_id
- campaign_exposed
- cart_total_value

【Variable SQL Expressions】
- cart_item_id: cart_items.cart_item_id
- campaign_exposed: CASE WHEN COUNT(campaign_exposure.exposure_id) > 0 THEN TRUE ELSE FALSE END
- cart_total_value: SUM(cart_items.price * cart_items.quantity)

【Schema】
TABLE: carts
  COLUMNS: cart_id (uuid), customer_id (uuid), created_at (date)
TABLE: cart_items
  COLUMNS: cart_item_id (uuid), cart_id (uuid), product_id (uuid), price (numeric), quantity (integer)
TABLE: campaign_exposure
  COLUMNS: exposure_id (uuid), cart_id (uuid), campaign_id (uuid)
TABLE: campaigns
  COLUMNS: campaign_id (uuid), name (text), start_date (date), end_date (date)

FOREIGN KEYS:
  carts.cart_id -> cart_items.cart_id
  carts.cart_id -> campaign_exposure.cart_id
  campaign_exposure.campaign_id -> campaigns.campaign_id

【SQL GENERATED】
```sql
WITH
campaign_exposed_per_cart AS (
    SELECT
        cart_id,
        CASE WHEN COUNT(exposure_id) > 0 THEN TRUE ELSE FALSE END AS campaign_exposed
    FROM campaign_exposure
    GROUP BY cart_id
),
final_table AS (
    SELECT
        cart_items.cart_item_id AS cart_item_id,
        SUM(cart_items.price * cart_items.quantity) AS cart_total_value
        campaign_exposed_per_cart.campaign_exposed AS campaign_exposed
    FROM cart_items 
    LEFT JOIN campaign_exposed_per_cart ON cart_items.cart_id = campaign_exposed_per_cart.cart_id
)
SELECT * FROM final_table;
```
【Output】
Step 1: cart_item_id, campaign_exposed, cart_total_value are all selected correctly using their corresponding SQL expressions.
Step 2: All joins are correct.
Step 3: The final query is valid and selects all required variables.
     
## Example2:
【Variables】
- student_id
- course_name
- professor_name
     
【Variable SQL Expressions】
- student_id: students.student_id
- course_name: courses.course_name
- professor_name: professors.name

【Schema】     
TABLE: students
  COLUMNS: student_id (uuid), name (text), birth_year (integer)
TABLE: enrollments
  COLUMNS: enrollment_id (uuid), student_id (uuid), course_id (uuid), enrolled_at (date)
TABLE: courses
  COLUMNS: course_id (uuid), course_name (text), professor_id (uuid)
TABLE: professors
  COLUMNS: professor_id (uuid), name (text)

FOREIGN KEYS:
  enrollments.student_id -> students.student_id
  enrollments.course_id -> courses.course_id
  courses.professor_id -> professors.professor_id
     
【SQL GENERATED】
```sql
WITH final_table as (
    SELECT
    students.student_id AS student_id,
    courses.course_name AS course_name,
    professors.name AS professor_name
FROM students
JOIN courses ON students.student_id = courses.course_id 
JOIN professors ON courses.professor_id = professors.professor_id)
SELECT * FROM final_table;
```
【Output】
Step 1: student_id, course_name, professor_name are all selected correctly using their corresponding SQL expressions.
Step 2: The join between students and courses is incorrect. It should be based on enrollments table. 
Step 3: The revised query is:
```sql
WITH final_table AS (
    SELECT
    students.student_id AS student_id,
    courses.course_name AS course_name,
    professors.name AS professor_name
FROM enrollments
JOIN students ON enrollments.student_id = students.student_id
JOIN courses ON enrollments.course_id = courses.course_id
JOIN professors ON courses.professor_id = professors.professor_id)
SELECT * FROM final_table;
```

## Example3:
【Variables】
- customer_id
- name
- account_balance
- monthly_deposit_total

【Variable SQL Expressions】
- customer_id: customers.customer_id
- name: customers.name
- account_balance: accounts.balance
- monthly_deposit_total: SUM(CASE WHEN transactions.transaction_type = ‘deposit’
AND DATE_TRUNC(‘month’, transactions.transaction_date) = DATE_TRUNC(‘month’, CURRENT_DATE)
THEN transactions.transaction_amount ELSE 0 END)

【Schema】
TABLE: customers
COLUMNS: customer_id (uuid), name (text), date_of_birth (date)
TABLE: accounts
COLUMNS: account_id (uuid), customer_id (uuid), balance (numeric), account_type (text)
TABLE: transactions
COLUMNS: transaction_id (uuid), account_id (uuid), transaction_amount (numeric), transaction_type (text), transaction_date (date)

FOREIGN KEYS:
accounts.customer_id -> customers.customer_id
transactions.account_id -> accounts.account_id
     
【SQL GENERATED】
```sql
WITH final_table AS (
    SELECT
    customers.customer_id AS customer_id,
    customers.name AS name,
    accounts.balance AS account_balance
FROM customers
JOIN accounts ON accounts.customer_id = customers.customer_id)
SELECT * FROM final_table;
```
【Output】
Step 1: customer_id, name, account_balance are selected correctly, but monthly_deposit_total is missing.
Step 2: The join between customers and accounts is correct, but transactions table is not included.
Step 3: The revised query is:
```sql
WITH monthly_deposits AS (
    SELECT
        accounts.customer_id,
        SUM(transactions.transaction_amount) AS monthly_deposit_total
    FROM accounts
    JOIN transactions ON transactions.account_id = accounts.account_id
    WHERE transactions.transaction_type = 'deposit'
      AND DATE_TRUNC('month', transactions.transaction_date) = DATE_TRUNC('month', CURRENT_DATE)
    GROUP BY accounts.customer_id
),
WITH final_table as (
    SELECT
    customers.customer_id AS customer_id,
    customers.name AS name,
    accounts.balance AS account_balance,
    COALESCE(monthly_deposits.monthly_deposit_total, 0) AS monthly_deposit_total
FROM customers
JOIN accounts ON accounts.customer_id = customers.customer_id
LEFT JOIN monthly_deposits ON monthly_deposits.customer_id = customers.customer_id)
SELECT * FROM final_table;
```
==============================
Now please review the following SQL query and check if it meets the requirements. 
"""),
    ("human", """
【Variables】
{selected_variables}

【Variable SQL Expressions】
{variable_expressions}

【Schema】
{table_schemas}

【SQL GENERATED】
```sql
{sql}
```
""")])

### 4. Fix SQL Prompt
fix_sql_prompt = ChatPromptTemplate.from_messages([
    ("system", """
An PostgreSQL was generated to fetch data including all variables indicated below for causal analysis, but it failed with an error.
Please fix up PostgeSQL code based on query and database info. Solve the task step by step if you need to.
     
You will be given:
1. The old PostgreSQL query
2. The error message it triggered
3. The list of variable names to select and corresponding SQL expressions. All variables must be selected in the final query.
4. Table schemas

## Rules (PostgreSQL Specific)
- Never use aggregate functions (MAX, MIN, SUM, etc.) directly on boolean columns or boolean expressions in PostgreSQL.
- Always cast boolean expressions to integer (e.g., (expression)::int) when using them inside aggregate functions.
- If a boolean value is needed in the final result, cast it back or compare the result (e.g., MAX((flag)::int) = 1).
- Example:
  - Incorrect:   MAX(flag_column)
  - Correct:     MAX((flag_column)::int)

Now please fixup old SQL and generate new SQL again. Only output the new SQL in the code block, and indicate script type by ```sql ```in the code block.
"""),
    ("human", """
Old SQL Query:
```sql
{original_sql}
```
PostgreSQL Error:
{error_message}
     
Variables to Select:
{graph_nodes}

Variable Expressions:
{expression_dict}

Table Schemas:
{table_schemas}
""")
])


### 5. Causal Strategy Selection Prompt
class SelectedCausalStrategy(BaseModel):
    causal_task: str = Field(..., description="Overall causal task (e.g., estimating_causal_effect, mediation_analysis, causal_prediction, what_if)")
    identification_strategy: str = Field(..., description="Strategy used to identify the causal effect, e.g., backdoor, frontdoor, iv, mediation")
    estimation_method: str = Field(..., description="Estimation method compatible with identification strategy")
    refutation_methods: List[str] = Field(..., description="Optional refutation methods to test robustness")

strategy_output_parser = PydanticOutputParser(pydantic_object=SelectedCausalStrategy)

causal_strategy_prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are a causal inference expert using the DoWhy library.

Your job is to choose the appropriate causal inference strategy, estimation method, and optional refutation methods given:
- A user's causal question
- Extracted variables (treatment, outcome, confounders, mediators, instrumental variables, colliders)
    - data type information as `{treatment_type}` and `{outcome_type}`
- Basic data preview

Only choose from the valid options defined below.

---
Causal Tasks:
- estimating_causal_effect: estimate average treatment effect (ATE)
- mediation_analysis: decompose effect via mediators
- causal_prediction: predict outcome using causal structure (e.g., TabPFN)
- what_if: simulate counterfactuals
- root_cause: identify potential causes of outcome

Identification Strategies:
- backdoor: adjust for confounders that affect both treatment and outcome
- frontdoor: identify effect using mediators when backdoor not available
- iv: use instrumental variables for exogenous variation
- mediation: isolate indirect vs direct effects
- id_algorithm: automated graphical ID algorithm

- If outcome_type is "binary", prefer using backdoor.generalized_linear_model
- If outcome_type is "continuous", prefer using backdoor.linear_regression

Estimation Methods:
- backdoor.linear_regression: basic OLS on adjusted data
- backdoor.propensity_score_matching: match units by treatment probability
- backdoor.propensity_score_stratification: stratify by score and estimate
- backdoor.propensity_score_weighting: reweight sample by inverse propensity
- backdoor.distance_matching: match using nearest-neighbor distance
- backdoor.generalized_linear_model: GLM for non-normal outcomes
- iv.instrumental_variable: two-stage least squares using IV
- iv.regression_discontinuity: exploit cutoff-based variation
- frontdoor.two_stage_regression: mediator-based 2-stage estimator
- mediation.two_stage_regression: mediation-specific 2-stage model
- causal_prediction.tabpfn: use TabPFN to predict causal outcomes
- what_if.simple_model: simulate counterfactual with regression
- what_if.tabpfn: simulate counterfactual using TabPFN

Refutation Methods (optional):
- placebo_treatment_refuter: randomly replace treatment and re-test
- random_common_cause: add synthetic common cause to check stability
- data_subset_refuter: re-run analysis on subsets
- add_unobserved_common_cause: simulate bias from unobserved variables

{format_instructions}
"""),
    ("human", """
User Causal Question:
{question}

Parsed Variables:
- Treatment: {treatment}
- Outcome: {outcome}
- Confounders: {confounders}
- Mediators: {mediators}
- Instrumental Variables: {instrumental_variables}
- Colliders: {colliders}

Data Sample:
{df_sample}

Treatment Type: {treatment_type}
Outcome Type: {outcome_type}
""")
]).partial(format_instructions=strategy_output_parser.get_format_instructions())

### 6. Causal Analysis Result Generation Prompt
class CausalResultExplanation(BaseModel):
    explanation: str = Field(..., description="Plain-text summary of the causal analysis results.")

generate_answer_parser = PydanticOutputParser(pydantic_object=CausalResultExplanation)

generate_answer_prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are a causal inference expert. Based on the following information, write a clear but informative explanation of the causal analysis results.

Causal task metadata:
- Task type: {task}
- Estimator used: {estimation_method}
- Estimated causal effect: {causal_effect_value}
- 95% confidence interval: {causal_effect_ci}
- Refutation result (if any): {refutation_result}
- Label mappings (optional): {label_maps}

Parsed query details:
- Treatment variable: {treatment} — {treatment_expression_description}
- Outcome variable: {outcome} — {outcome_expression_description}
- Confounders: {confounders}
- Mediators (if any): {mediators}
- Instrumental variables (if any): {instrumental_variables}

Your goal is to make the result interpretable to a data-literate but non-expert audience. Add an example or intuitive description where possible.

Your explanation should include:
1. A plain interpretation of the estimated causal effect in everyday language. 
   - Along with statistical language, provide a translation of directionality (positive/negative effect) into intuitive language (e.g., "users who signed up longer ago are slightly less active").
2. Whether the effect is statistically significant based on p-value or CI,
3. Whether the refutation result strengthens or weakens confidence in the finding,
4. Any caveats or assumptions that should be kept in mind,
5. If label mappings are provided, interpret the effect in human terms.

Respond with a concise analytical summary (3–6 sentences).

{format_instructions}
"""),
    ("human", "")
]).partial(format_instructions=generate_answer_parser.get_format_instructions())