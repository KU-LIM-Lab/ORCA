import re
from typing import Dict
from collections import defaultdict

def extract_table_metadata(table_markdowns):
    table_metadata = defaultdict(set)
    primary_keys = set()

    for markdown in table_markdowns:
        lines = markdown.split('\n')
        current_table = None

        for line in lines:
            # extract table names
            if line.startswith("# Table: "):
                current_table = line.replace("# Table: ", "").strip()

            # extract PKs
            elif "PK" in line:
                match = re.match(r"- (\w+) \([^)]+\), PK", line)
                if match and current_table:
                    col = match.group(1)
                    primary_keys.add(f"{current_table}.{col}")
                    table_metadata[current_table].add(col)

            # extract regular columns
            elif line.startswith("- ") and current_table:
                col = line.split()[1]
                col = col.strip("()")
                if col not in ["FOREIGN", "KEY:"]:
                    table_metadata[current_table].add(col)

    return table_metadata, primary_keys

def validate_parsed_query(parsed_query, table_metadata, primary_keys):
    issues = []
    check_fields = ['treatment', 'outcome'] + parsed_query['confounders'] + parsed_query['mediators'] + parsed_query['instrumental_variables']

    for field in check_fields:
        if isinstance(field, str):
            # skip SQL expressions
            if '.' not in field:
                continue

            table, col = field.split('.')
            col_set = table_metadata.get(table, set())

            full_col = f"{table}.{col}"

            # non-existent column
            if col not in col_set:
                issues.append(f"{field} is not a valid column in {table} table.")

            # identifier-like variable
            if (
                col.endswith("_id") or 
                full_col in primary_keys
            ):
                issues.append(f"Do not use an identifier-like variable {field} in the parsed query. It should not be used as a causal variable.")

    return issues

def _is_identifier_like(var: str, table_metadata: Dict[str, set], primary_keys: set) -> bool:
    if '.' not in var:
        return False
    table, col = var.split('.')
    full_col = f"{table}.{col}"

    # non-existent column
    if col not in table_metadata.get(table, set()):
        return True

    # identifier-like (e.g., ends with _id or is a PK)
    if col.endswith("_id") or full_col in primary_keys:
        return True

    return False

def sanitize_parsed_query(parsed_query: Dict, table_metadata=None, primary_keys=None) -> Dict:
    """
    Remove confounders that overlap with treatment or outcome;
    optionally remove identifier-like or non-existent variables.
    """
    treatment = parsed_query.get("treatment")
    outcome = parsed_query.get("outcome")

    for key in ["confounders", "mediators", "instrumental_variables"]:
        vars_list = parsed_query.get(key, [])
        if isinstance(vars_list, list):
            # remove variables overlapping with treatment/outcome
            cleaned_vars = [v for v in vars_list if v not in {treatment, outcome}]

            # remove identifier-like / non-existent columns
            if table_metadata and primary_keys:
                cleaned_vars = [
                    v for v in cleaned_vars
                    if not _is_identifier_like(v, table_metadata, primary_keys)
                ]

            parsed_query[key] = cleaned_vars
    return parsed_query