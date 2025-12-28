# selector_template_for_causal = """
# As an expert in Causal Inference and Database Schema, your task is to analyze the User Question and Database Schema to prepare a dataset for **Causal Discovery**.

# Your goal is two-fold:
# 1. Identify Causal Variables: Extract the Treatment (cause) and Outcome (effect) variables from the user's question and map them to specific table columns.
# 2. Select Contextual Data: Identify relevant tables and columns to build a "flat table" that includes T, Y, and Potential Confounders (static attributes in parent tables).

# [Instruction]:
# Step 1: Causal Role Identification
# - Analyze the query to determine what is being manipulated (Treatment) and what is being measured (Outcome).
# - Map these to the most appropriate columns in the `{desc_str}`.

# Step 2: Table & Column Scouting (for Confounders)
# - IMPORTANT: For the table(s) containing the Treatment and Outcome variables, set the table decision to "keep_all".
# - Also explicitly return the Treatment and Outcome mappings in the output JSON (see format below).
# - Identify the table(s) containing Treatment and Outcome.
# - Trace Foreign Keys: Look for "Parent Tables" (e.g., User, Store, Product) connected to the T/Y tables.
# - Include Parent Tables: Select these tables as they likely contain confounders (e.g., user demographics, store region).
# - Filter Columns: - Keep "Attribute" columns (categorical or numerical features like age, status, price).
#     - Drop high-cardinality IDs (except join keys), raw text descriptions, or sensitive PII unless critical.
#     - Mark tables as "keep_all" if most columns are useful features.
#     - Mark irrelevant tables (logs, system configs) as "drop_all".

# [ABSOLUTE OUTPUT RULES — DO NOT VIOLATE]
# 1) You MUST output a JSON object with EXACT keys: "selected_schema", "treatment", "outcome".
# 2) "selected_schema" MUST be a mapping: {{ "<table_name>": "keep_all" | "drop_all" | ["col1","col2",...] }}.
# 3) CRITICAL: The table that contains the Treatment column MUST be "keep_all".
# 4) CRITICAL: The table that contains the Outcome column MUST be "keep_all".
#    - Even if you think only a few columns are needed, you must still output "keep_all" for those tables.
# 5) Do NOT replace "keep_all" with a list of columns for Treatment/Outcome tables.
# 6) If you violate this format, the system will treat the answer as incorrect.

# Here is a typical example:

# ==========
# 【DB_ID】 banking_system
# 【Schema】
# # Table: account
# Table description: Contains information about bank accounts, including account ID, district ID, frequency, and creation date.
# [
#   (account_id, the id of the account. Value examples: [11382, 11362].),
#   (district_id, location of branch. Value examples: [77, 76].),
#   (frequency, frequency of the acount. Value examples: ['POPLATEK MESICNE', 'POPLATEK TYDNE'].),
#   (date, the creation date of the account. Value examples: ['1997-12-29', '1997-12-28'].)
# ]
# # Table: client
# Table description: Contains personal information about clients.
# [
#   (client_id, the unique number. Value examples: [13998, 13971].),
#   (gender, gender. Value examples: ['M', 'F'].),
#   (birth_date, birth date. Value examples: ['1987-09-27', '1986-08-13'].),
#   (district_id, location of branch. Value examples: [77, 76].)
# ]
# # Table: loan
# Table description: Contains information about loans that clients have taken.
# [
#   (loan_id, the id number identifying the loan data. Value examples: [4959, 4960].),
#   (account_id, the id number identifying the account. Value examples: [10, 80].),
#   (date, the date when the loan is approved. Value examples: ['1998-07-12'].),
#   (amount, the id number identifying the loan data. Value examples: [1567, 7877].),
#   (duration, the id number identifying the loan data. Value examples: [60, 48].),
#   (payments, the id number identifying the loan data. Value examples: [3456, 8972].),
#   (status, the id number identifying the loan data. Value examples: ['C', 'A'].)
# ]
# # Table: district
# Table description: Contains information about districts.
# [
#   (district_id, location of branch. Value examples: [77, 76].),
#   (A4, number of inhabitants. Value examples: [95907, 95616].),
#   (A11, average salary. Value examples: [12541, 11277].),
#   (A12, poverty rate. Value examples: [12.4, 9.8].),
#   (A13, unemployment rate. Value examples: [8.2, 7.9].)
# ]
# 【Foreign keys】
# client."district_id" = district."district_id"
# account."district_id" = district."district_id"
# loan."account_id" = account."account_id"
# client."client_id" (implied via account owner mapping, conceptually linked)

# 【Question】
# How does the loan duration affect the approved loan amount?
# 【Evidence】
# 'duration' in table loan represents the term; 'amount' represents the principal. Socio-economic factors (district info) and demographics (client info) might influence both.

# 【Answer】
# ```json
# {{
#   "selected_schema": {{
#     "loan": "keep_all",
#     "account": ["frequency", "date"],
#     "client": ["gender", "birth_date"],
#     "district": ["A11", "A12", "A13", "A4"]
#   }},
#   "treatment": {{"table": "loan", "column": "duration"}},
#   "outcome": {{"table": "loan", "column": "amount"}}
# }}
# ```
# Question Solved.


# ==========

# Here is a new example, please start answering:

# 【DB_ID】 {db_id}
# 【Schema】
# {desc_str}
# 【Foreign keys】
# {fk_str}
# 【Question】
# {query}
# 【Evidence】
# {evidence}
# 【Answer】
# """

selector_template_for_causal = """
As an expert in Causal Inference and Database Schema, your task is to analyze the User Question and Database Schema to prepare a dataset for **Causal Discovery**.

Your goal is two-fold:
1. Identify Causal Variables: Extract the Treatment (cause) and Outcome (effect) variables from the user's question and map them to specific table columns.
2. Select Contextual Data: Identify relevant tables and columns to build a "flat table" that includes T, Y, and Potential Confounders (static attributes in parent tables).

[Instruction]:
Step 1: Causal Role Identification
- Analyze the query to determine what is being manipulated (Treatment) and what is being measured (Outcome).
- Map these to the most appropriate columns in the `{desc_str}`.

Step 2: Table & Column Scouting (for Confounders)
- IMPORTANT: Treatment and Outcome variables, Must be selected.
- Also explicitly return the Treatment and Outcome mappings in the output JSON (see format below).
- Identify the table(s) containing Treatment and Outcome.
- Trace Foreign Keys: Look for "Parent Tables" (e.g., User, Store, Product) connected to the T/Y tables.
- Include Parent Tables: Select these tables as they likely contain confounders (e.g., user demographics, store region).
- Filter Columns: - Keep "Attribute" columns (categorical or numerical features like age, status, price).
    - Drop high-cardinality IDs (except join keys), raw text descriptions, or sensitive PII unless critical.
    - Mark tables as "keep_all" if most columns are useful features.
    - Mark irrelevant tables (logs, system configs) as "drop_all".

[ABSOLUTE OUTPUT RULES — DO NOT VIOLATE]
1) You MUST output a JSON object with EXACT keys: "selected_schema", "treatment", "outcome".
2) "selected_schema" MUST be a mapping: {{ "<table_name>": ["col1","col2",...] }}.
3) CRITICAL: The table that contains the Treatment column MUST be included.
4) CRITICAL: The table that contains the Outcome column MUST be included.
6) If you violate this format, the system will treat the answer as incorrect.

Here is a typical example:

==========
【DB_ID】 banking_system
【Schema】
# Table: account
Table description: Contains information about bank accounts, including account ID, district ID, frequency, and creation date.
[
  (account_id, the id of the account. Value examples: [11382, 11362].),
  (district_id, location of branch. Value examples: [77, 76].),
  (frequency, frequency of the acount. Value examples: ['POPLATEK MESICNE', 'POPLATEK TYDNE'].),
  (date, the creation date of the account. Value examples: ['1997-12-29', '1997-12-28'].)
]
# Table: client
Table description: Contains personal information about clients.
[
  (client_id, the unique number. Value examples: [13998, 13971].),
  (gender, gender. Value examples: ['M', 'F'].),
  (birth_date, birth date. Value examples: ['1987-09-27', '1986-08-13'].),
  (district_id, location of branch. Value examples: [77, 76].)
]
# Table: loan
Table description: Contains information about loans that clients have taken.
[
  (loan_id, the id number identifying the loan data. Value examples: [4959, 4960].),
  (account_id, the id number identifying the account. Value examples: [10, 80].),
  (date, the date when the loan is approved. Value examples: ['1998-07-12'].),
  (amount, the id number identifying the loan data. Value examples: [1567, 7877].),
  (duration, the id number identifying the loan data. Value examples: [60, 48].),
  (payments, the id number identifying the loan data. Value examples: [3456, 8972].),
  (status, the id number identifying the loan data. Value examples: ['C', 'A'].)
]
# Table: district
Table description: Contains information about districts.
[
  (district_id, location of branch. Value examples: [77, 76].),
  (A4, number of inhabitants. Value examples: [95907, 95616].),
  (A11, average salary. Value examples: [12541, 11277].),
  (A12, poverty rate. Value examples: [12.4, 9.8].),
  (A13, unemployment rate. Value examples: [8.2, 7.9].)
]
【Foreign keys】
client."district_id" = district."district_id"
account."district_id" = district."district_id"
loan."account_id" = account."account_id"
client."client_id" (implied via account owner mapping, conceptually linked)

【Question】
How does the loan duration affect the approved loan amount?
【Evidence】
'duration' in table loan represents the term; 'amount' represents the principal. Socio-economic factors (district info) and demographics (client info) might influence both.

【Answer】
```json
{{
  "selected_schema": {{
    "loan": ["duration","amount", "payments", "status"],
    "account": ["frequency", "date"],
    "client": ["gender", "birth_date"],
    "district": ["A11", "A12", "A13", "A4"]
  }},
  "treatment": {{"table": "loan", "column": "duration"}},
  "outcome": {{"table": "loan", "column": "amount"}}
}}
```
Question Solved.


==========

Here is a new example, please start answering:

【DB_ID】 {db_id}
【Schema】
{desc_str}
【Foreign keys】
{fk_str}
【Question】
{query}
【Evidence】
{evidence}
【Answer】
"""


# # Need to modify if using another DB
# decompose_template_for_causal = """
# Given a 【Database schema】, a knowledge 【Evidence】, the 【Question】, and the 【Selected Variables】 from the previous step, your task is to generate ONE single PostgreSQL query that constructs a flat dataset for Causal Discovery.

# your goal here is to retrieve a wide dataset containing the Treatment, Outcome, and all Potential Confounders while preserving the data population.

# 【Constraints】
# - Main Table: Start `FROM` the table containing the Treatment variable (or the central transaction table).
# - Join Strategy: Use **`LEFT JOIN`** for all related tables to prevent data loss (missing rows) unless you are certain an INNER JOIN is required to remove invalid data.
# - Column Selection: Select ALL variables listed in 【Selected Variables】.
# - Aliasing: If column names collide (e.g., `id`, `created_at` appear in multiple tables), rename them clearly (e.g., `user_created_at`, `order_created_at`).
# - No Aggregation: Do not use GROUP BY unless a specific feature requires aggregation (e.g., counting orders per user). We generally need row-level data.
# - Output format : When you write SQL query, you must wrap code with ```sql ...```

# ==========

# 【Database schema】
# # Table: account
# [("account_id", "id".), ("district_id", "location".), ("date", "creation date".)]
# # Table: client
# [("client_id", "id".), ("gender", "gender".), ("birth_date", "birth date".), ("district_id", "location".)]
# # Table: loan
# [("loan_id", "id".), ("account_id", "account link".), ("amount", "loan amount".), ("duration", "loan duration".)]
# # Table: district
# [("district_id", "id".), ("A11", "average salary".), ("A4", "population".)]

# 【Foreign keys】
# account."district_id" = district."district_id"
# client."district_id" = district."district_id"
# loan."account_id" = account."account_id"

# 【Question】
# How does the loan duration affect the approved loan amount?

# 【Selected Variables】
# - Treatment: loan.duration
# - Outcome: loan.amount
# - Confounders: client.gender, client.birth_date, district.A11, district.A4

# Decompose the task into construction steps, considering 【Constraints】, and generate the SQL:

# Step 1: Identify the Main Table (Unit of Analysis).
# - The Treatment (`duration`) and Outcome (`amount`) are in the `loan` table. This is our base.

# Step 2: Plan the Joins to reach Confounders.
# - `loan` -> `account` (via account_id)
# - `account` -> `client` (via client_id logic, usually linked conceptually or via disposition, here assuming direct link for example) -> `district` (via district_id).
# - Use LEFT JOINs to keep all loans.

# Step 3: Construct the Final SQL.
# ```sql
# SELECT 
#     T1."duration" AS treatment_duration,
#     T1."amount" AS outcome_amount,
#     T3."gender",
#     T3."birth_date",
#     T4."A11" AS district_avg_salary,
#     T4."A4" AS district_population
# FROM loan AS T1
# LEFT JOIN account AS T2 ON T1."account_id" = T2."account_id"
# LEFT JOIN client AS T3 ON T2."account_id" = T3."client_id" -- Assuming link exists for example
# LEFT JOIN district AS T4 ON T3."district_id" = T4."district_id";
# ```
# Dataset Construction Solved.

# ==========

# 【Database schema】
# Table: frpm
# [ ("CDSCode", "School Code".), ("Charter School (Y/N)", "Is Charter?".), ("Free Meal Count", "Poverty Proxy".) ]
# Table: satscores
# [ ("cds", "School Code".), ("AvgScrMath", "Math Score".), ("NumTstTakr", "Number of Takers".) ] 

# 【Foreign keys】 
# frpm."CDSCode" = satscores."cds"

# 【Question】 
# Does being a Charter School affect SAT Math scores? 【Selected Variables】

# 【Selected Variables】
# - Treatment: frpm."Charter School (Y/N)"
# - Outcome: satscores."AvgScrMath"
# - Confounders: frpm."Free Meal Count", satscores."NumTstTakr"

# Decompose the task into construction steps, considering 【Constraints】, and generate the SQL:

# Step 1: Identify the Main Table.
# - Treatment is in frpm (School characteristics). Outcome is in satscores. We can start with frpm as the base of schools.

# Step 2: Plan the Joins.
# - frpm -> satscores using CDSCode.
# - Use LEFT JOIN to include schools even if they don't have SAT scores (though for analysis we might filter nulls later, strictly data fetching should be inclusive).

# Step 3: Construct the Final SQL.

# ```sql
# SELECT 
#     T1."Charter School (Y/N)" AS is_charter,
#     T2."AvgScrMath" AS math_score,
#     T1."Free Meal Count",
#     T2."NumTstTakr"
# FROM frpm AS T1
# LEFT JOIN satscores AS T2 ON T1."CDSCode" = T2."cds";
# ```
# Dataset Construction Solved.

# ==========

# 【Database schema】 {desc_str} 
# 【Foreign keys】 {fk_str} 
# 【Question】 {query} 
# 【Selected Variables】 {selected_info} 
# 【Evidence】 {evidence}

# Decompose the task into construction steps, considering 【Constraints】, and generate the SQL: 
# """
# Need to modify if using another DB
decompose_template_for_causal = """
Given a 【Database schema】, a knowledge 【Evidence】, the 【Question】, and the 【Selected Variables】 from the previous step, your task is to generate ONE single PostgreSQL query that constructs a flat dataset for Causal Discovery.

your goal here is to retrieve a wide dataset containing the Treatment, Outcome, and all Potential Confounders while preserving the data population.

- Key Column Normalization (Avoid Duplicate IDs):
  - If the same logical key appears in multiple joined tables (e.g., `user_id`, `coupon_id`, `order_id`) and the join path implies they should match,
    produce ONE canonical column using `COALESCE(left_table.key, right_table.key)` and alias it as the canonical name (e.g., `user_id`).
  - Keep table-specific raw key columns ONLY if they can legitimately differ; otherwise do not output duplicate key columns with the same name.
  - Consistency check: If both sides are non-NULL but could differ, preserve both as separate aliased columns (e.g., `user_id_from_T1`, `user_id_from_T3`)
    so downstream code can detect conflicts. Do NOT silently drop rows or use INNER JOIN to “fix” inconsistencies.

【Constraints】
- Main Table: Start `FROM` the table containing the Treatment variable (or the central transaction table).
- Join Strategy: Use **`LEFT JOIN`** for all related tables to prevent data loss (missing rows) unless you are certain an INNER JOIN is required to remove invalid data.
- Column Selection: Select ALL variables listed in 【Selected Variables】.
  - For tables marked "keep_all", include all columns from that table (e.g., `T1.*`).
  - For tables with a column list, include ONLY those columns.
  - Do NOT select foreign-key columns solely used for JOIN conditions unless they are explicitly listed in the selected_schema.
- Aliasing: If column names collide (e.g., `id`, `created_at` appear in multiple tables), rename them clearly (e.g., `user_created_at`, `order_created_at`).
- No Aggregation: Do not use GROUP BY unless a specific feature requires aggregation (e.g., counting orders per user). We generally need row-level data.
- Output format : When you write SQL query, you must wrap code with ```sql ...```

==========

【Foreign keys】
account."district_id" = district."district_id"
client."district_id" = district."district_id"
loan."account_id" = account."account_id"

【Question】
How does the loan duration affect the approved loan amount?

【Selected Variables】 (table-level selection)
```json
{{
  "selected_schema": {{
    "loan": "keep_all",
    "account": ["district_id", "date"],
    "client": ["gender", "birth_date", "district_id"],
    "district": ["A11", "A4"]
  }},
  "treatment": {{"table": "loan", "column": "duration"}},
  "outcome": {{"table": "loan", "column": "amount"}}
}}
```

Decompose the task into construction steps, considering 【Constraints】, and generate the SQL:

Step 1: Identify the Main Table (Unit of Analysis).
	•	Treatment and Outcome are in loan, so start FROM loan.

Step 2: Plan the Joins.
	•	loan -> account via account_id
	•	client join path is domain-specific; for this example assume client is reachable from account (placeholder).
	•	account and client -> district via district_id
	•	Use LEFT JOINs.

Step 3: Construct the Final SQL.
```sql
SELECT
    T1.*,
    T2."district_id" AS account_district_id,
    T2."date" AS account_date,
    T3."gender" AS client_gender,
    T3."birth_date" AS client_birth_date,
    T3."district_id" AS client_district_id,
    T4."A11" AS district_A11,
    T4."A4" AS district_A4
FROM loan AS T1
LEFT JOIN account AS T2 ON T1."account_id" = T2."account_id"
LEFT JOIN client  AS T3 ON T2."account_id" = T3."client_id"  -- placeholder join for example only
LEFT JOIN district AS T4 ON T2."district_id" = T4."district_id";
```
Dataset Construction Solved.

==========


【Foreign keys】 
frpm."CDSCode" = satscores."cds"

【Question】 
Does being a Charter School affect SAT Math scores?

【Selected Variables】 (table-level selection)
```json
{{
  "selected_schema": {{
    "frpm": "keep_all",
    "satscores": ["AvgScrMath", "NumTstTakr"]
  }},
  "treatment": {{"table": "frpm", "column": "Charter School (Y/N)"}},
  "outcome": {{"table": "satscores", "column": "AvgScrMath"}}
}}
```

Decompose the task into construction steps, considering 【Constraints】, and generate the SQL:

Step 1: Identify the Main Table.
	•	Treatment is in frpm, so start FROM frpm.

Step 2: Plan the Joins.
	•	frpm -> satscores via CDSCode/cds using LEFT JOIN.

Step 3: Construct the Final SQL.

```sql
SELECT
    T1.*,
    T2."AvgScrMath" AS satscores_AvgScrMath,
    T2."NumTstTakr" AS satscores_NumTstTakr
FROM frpm AS T1
LEFT JOIN satscores AS T2 ON T1."CDSCode" = T2."cds";
```
Dataset Construction Solved.

==========

【Foreign keys】 {fk_str} 
【Question】 {query} 
【Selected Variables】 {selected_info} 
【Evidence】 {evidence}

Decompose the task into construction steps, considering 【Constraints】, and generate the SQL: 
"""

refiner_template_for_causal = """
【Instruction】
When executing SQL below, some errors occurred, please fix up SQL based on query and database info.
Solve the task step by step if you need to.
When you find an answer, verify the answer carefully. Include verifiable evidence in your response if possible.

【Constraints】
- Preserve Columns: Ensure that Treatment, Outcome, and ALL selected Confounder columns are present in the `SELECT` clause.
- Fix Ambiguity: If the error is "ambiguous column name", add the table alias (e.g., `T1.created_at`) to the column.
- Join Strategy: Prefer `LEFT JOIN` to preserve the main population (rows). Use `INNER JOIN` only if necessary to filter invalid data.
- No Aggregation: Do NOT use aggregate functions (`AVG`, `COUNT`, `MAX`) in the final SELECT unless the feature itself is an aggregate (e.g., "total_orders_per_user"). We need row-level data.
- Data Types: Ensure compatible data types when joining or comparing.

【Query】
-- {query}
【Evidence】
{evidence}
【Database info】
{desc_str}
【Foreign keys】
{fk_str}
【old SQL】
```sql
{sql}
```
【POSTGRESQL error】 
{sql_error}
【Exception class】
{exception_class}

Now please fixup old SQL and generate new SQL again. Only output the new SQL in the code block, and indicate script type by ```sql ```in the code block.
【correct SQL】
"""

refiner_feedback_template_for_causal = """
【Instruction】
When executing SQL below, no rows were returned. please fix up SQL based on query, database info, and feedback on the old SQL.
Solve the task step by step if you need to.
When you find an answer, verify the answer carefully. Include verifiable evidence in your response if possible.

【Constraints】
- Handle "No Rows": If the feedback says "No rows returned", check your JOIN conditions. 
  - Change `INNER JOIN` to `LEFT JOIN` if strict matching is filtering out all data.
  - Check `WHERE` clauses for overly strict filters.
- Handle "Missing Columns": If the feedback says a variable is missing, add it to the `SELECT` clause with a proper alias.
- Handle "Row Explosion": If the feedback says "Too many rows" (1:N problem), ensure you are joining on the correct Primary Key/Foreign Key. You might need to pre-aggregate the joined table (e.g., take the `MAX` or `AVG` of the confounder) before joining.
- **Maintain Structure:** The final output must be a single SELECT statement.
【Query】
-- {query}
【Evidence】
{evidence}
【Database info】
{desc_str}
【Foreign keys】
{fk_str}

【Feedback】
{review_feedback}
【old SQL】
```sql
{sql}
```

Now please fixup old SQL and generate new SQL again. Only output the new SQL in the code block, and indicate script type by ```sql ```in the code block.
【correct SQL】
"""


review_noresult_template_for_causal = """
You generated the following SQL for a user's question. The query executed, but no rows were returned.

Your job is to determine:
- Is the SQL logically correct, meaning the structure and filter logic are appropriate for the user's question?
- Or is there a mistake in the SQL that, if corrected, would likely return results?

[Instructions]
- Check JOINs: Did the query use `INNER JOIN` on a sparse table (e.g., a specific survey or event log)? This often filters out the entire population. It should likely be a `LEFT JOIN`.
- Check Filters: Are the `WHERE` clauses mutually exclusive or too strict for the available data?
- Response Rule:
    - If the SQL correctly uses `LEFT JOIN`s to preserve data and seems logically sound, respond with: `Yes.` (It's a data availability issue).
    - If the SQL uses `INNER JOIN` risks dropping the population or has incorrect link logic, respond with: `No. <brief explanation>`

-------

Examples:

1.  
User Question: "How does the 'Premium' subscription status affect 'Time Spent'?"
SQL:  
SELECT T1.user_id, T2.is_premium, T1.time_spent
FROM users AS T1
LEFT JOIN subscriptions AS T2 ON T1.user_id = T2.user_id;
Output: Yes.
(The SQL uses LEFT JOIN, preserving all users even if they don't have a subscription. If 0 rows returned, the 'users' table must be empty. The logic is sound.)

2. 
User Question: "Does the 'Email Campaign' increase 'Purchase Amount'?"
SQL:
SELECT T1.user_id, T2.campaign_name, T1.amount
FROM purchases AS T1
INNER JOIN email_logs AS T2 ON T1.user_id = T2.user_id
WHERE T2.campaign_date > '2024-01-01';
Output: No. Using INNER JOIN restricts the dataset ONLY to users who received an email AND made a purchase. This eliminates the control group (users who didn't get email) and might return 0 rows if the date condition is too strict. It should be a LEFT JOIN starting from the population table.

--------

Now analyze the following SQL and answer using the same format. Use the shema information provided to determine if the SQL logic is sound.:

User Question: {query}
SQL: {sql}
schema info: {desc_str}
evidence: {evidence}

Output Format:
Yes.
No. <explanation of the logical issue>

Note: Do NOT say 'No.' if the SQL logic is sound but the data may be missing. That should be answered with 'Yes.'
"""


review_result_template_for_causal = """
You generated a SQL query to construct a **Causal Analysis Dataset**, and it executed successfully.
Your job is to determine if the returned dataset is suitable for Causal Discovery algorithms (like PC or GES).

Input:
- User Question
- SQL Result
  (Note: it contains the row count and a sample of the data)

Quality Checklist for Causal Data:
1. No Aggregation (Crucial): Did the query return only 1 row (or very few)? 
   - If yes, the SQL likely calculated an average/sum instead of retrieving raw data. This is **Insufficient**.
   - We need **row-level data** (many rows representing individual users, transactions, etc.).
2. Sufficient Sample Size: Is the row count < 10? 
   - If yes, it's too small for statistical inference. This is Insufficient.
3. Data Variance: Does the sample show variation in the Treatment column?
   - If the Treatment column has only one unique value (e.g., all are '1'), we cannot learn causality. Ideally, answer 'Yes' but note the risk, or 'No' if it looks like a logic error.
4. Missing Values: Are key columns (Treatment/Outcome) entirely NULL?

Output Format:
- If the dataset looks like a valid, multi-row flat table: `Yes.`
- If the dataset is aggregated, too small, or empty: `No. <explanation of the issue>`

-------
**Examples:**

1.
User Question: "Effect of discount on sales."
SQL Result: `[(0.15, 54000)]` (Row count: 1)
Output: No. The result is aggregated (single row). Causal discovery requires raw row-level data (e.g., one row per transaction), not the average discount and total sales.

2.
User Question: "Does studying time affect test scores?"
SQL Result: `[(10, 90), (5, 70), (8, 85), ...]` (Row count: 500)
Output: Yes.

3.
User Question: "Impact of ads on clicks."
SQL Result: `[(1, 0), (1, 0), (1, 5), (1, 2)...]` (Row count: 1000, but 'Ad_Shown' column is always 1)
Output: No. The treatment variable 'Ad_Shown' has no variation (all 1). We need data from both users who saw ads AND users who didn't (Control group). Check the WHERE clause.
-------

User question: {query}
SQL result: {result}

Output Format:
Yes.
No. <explanation of why the answer is insufficient>
"""
