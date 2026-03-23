"""
Baseline LLM prompt templates for data exploration experiments.

Intentionally simple — these are plain-LLM baselines, not agent-assisted.
"""

# ── Table Explorer ─────────────────────────────────────────────────────────────
# Simple: just ask for a detailed table description.

TABLE_EXPLORER_SYSTEM = "You are a helpful data analyst."

TABLE_EXPLORER_HUMAN = """Please provide a detailed description of the following database table.

Table: {table_name}

Schema:
{schema_markdown}

Sample Rows (up to {sample_limit} rows):
{sample_rows}"""


# ── Table Recommender ──────────────────────────────────────────────────────────
# Simple: given a user query, return the tables needed for the analysis.

TABLE_RECOMMENDER_SYSTEM = "You are a helpful data analyst."

TABLE_RECOMMENDER_HUMAN = """Which tables from the database are needed to perform the following analysis?

Analysis request: {question}

Available tables:
{table_list}

Return only a JSON array of the required table names. Example: ["table1", "table2"]
No explanation — just the JSON array."""


# ── Text2SQL ───────────────────────────────────────────────────────────────────
# Aligned with mini_dev/llm/src/prompt.py:
#   schema DDL → SQL-comment question block → CoT instruction → output instruction

TEXT2SQL_SYSTEM = "You are an expert SQL developer."

TEXT2SQL_HUMAN = """{schema_ddl}

-- Using valid PostgreSQL, answer the following questions for the tables provided above.
-- {question}{evidence_block}

Generate the PostgreSQL for the above question after thinking step by step:

In your response, you do not need to mention your intermediate steps.
Do not include any comments in your response.
Do not need to start with the symbol ```
You only need to return the result PostgreSQL SQL code starting from SELECT"""
