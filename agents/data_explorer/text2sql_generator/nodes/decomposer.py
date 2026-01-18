import re
from utils.llm import call_llm
from prompts.text2sql_generator_prompts import decompose_template
from prompts.text2sql_for_causal_prompts import decompose_template_for_causal

from langchain_core.language_models.chat_models import BaseChatModel

import re
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional, Set

FK_PATTERN = re.compile(
    r'^\s*(?P<t1>\w+)\."(?P<c1>[^"]+)"\s*=\s*(?P<t2>\w+)\."(?P<c2>[^"]+)"\s*$'
)

def parse_fk_str(fk_str: str) -> List[Tuple[str, str, str, str]]:
    """
    Parse lines like:
      user_coupons."coupon_id" = coupon."coupon_id"
    into tuples:
      (user_coupons, coupon_id, coupon, coupon_id)
    """
    edges = []
    for line in fk_str.splitlines():
        line = line.strip()
        if not line:
            continue
        m = FK_PATTERN.match(line)
        if not m:
            # skip unparsable lines (or raise)
            continue
        t1, c1, t2, c2 = m.group("t1"), m.group("c1"), m.group("t2"), m.group("c2")
        edges.append((t1, c1, t2, c2))
    return edges

def build_join_graph(fk_edges: List[Tuple[str, str, str, str]]):
    """
    Undirected adjacency for path search, but keep ON condition for each directed hop.
    We'll store both directions with swapped columns.
    """
    adj = defaultdict(list)
    for t1, c1, t2, c2 in fk_edges:
        # t1 -> t2
        adj[t1].append((t2, c1, c2))
        # t2 -> t1 (reverse)
        adj[t2].append((t1, c2, c1))
    return adj

def bfs_path(adj, start: str, goal: str) -> Optional[List[Tuple[str, str, str]]]:
    """
    Find a path from start to goal.
    Return list of hops: [(next_table, left_col, right_col), ...]
    meaning: current."left_col" = next."right_col"
    """
    if start == goal:
        return []

    q = deque([start])
    parent = {start: None}
    parent_edge = {}  # child -> (parent_table, left_col, right_col)

    while q:
        cur = q.popleft()
        for nxt, left_col, right_col in adj.get(cur, []):
            if nxt in parent:
                continue
            parent[nxt] = cur
            parent_edge[nxt] = (cur, left_col, right_col)
            if nxt == goal:
                q.clear()
                break
            q.append(nxt)

    if goal not in parent:
        return None

    # reconstruct
    hops_rev = []
    node = goal
    while node != start:
        cur, left_col, right_col = parent_edge[node]
        # hop from cur -> node
        hops_rev.append((node, left_col, right_col))
        node = cur
    hops_rev.reverse()
    return hops_rev

def choose_main_table(selected_schema: Dict[str, object]) -> str:
    """
    Default main table: the first table marked keep_all, else first table in dict.
    """
    for t, v in selected_schema.items():
        if isinstance(v, str) and v.lower() == "keep_all":
            return t
    return next(iter(selected_schema.keys()))

def generate_flat_dataset_sql(
    selected_schema: Dict[str, object],
    fk_str: str,
    main_table: Optional[str] = None,
    schema_name: Optional[str] = None,   # e.g., "public"
) -> str:
    """
    Rule-based wide flat dataset SQL generator (PostgreSQL).
    - LEFT JOIN only
    - Join paths derived only from fk_str
    - Aliasing for explicitly listed columns
    - keep_all tables use T?.* (without schema metadata we can't expand safely)
    """
    fk_edges = parse_fk_str(fk_str)
    adj = build_join_graph(fk_edges)

    if not main_table:
        main_table = choose_main_table(selected_schema)

    tables_to_select = list(selected_schema.keys())
    tables_needed = list(dict.fromkeys([main_table] + tables_to_select))

    # Build join plan: collect required joins along all shortest paths
    # Each join is represented as (left_table, right_table, left_col, right_col)
    joins: List[Tuple[str, str, str, str]] = []
    joined: Set[str] = {main_table}

    # We accumulate unique edges to avoid duplicates
    join_edges_set: Set[Tuple[str, str, str, str]] = set()

    for target in tables_needed:
        if target == main_table:
            continue
        path = bfs_path(adj, main_table, target)
        if path is None:
            # No path in FK graph: skip (or raise)
            # Here: raise to force correctness
            raise ValueError(f"No FK join path from '{main_table}' to '{target}' using provided fk_str")

        cur = main_table
        for (nxt, left_col, right_col) in path:
            edge = (cur, nxt, left_col, right_col)
            if edge not in join_edges_set:
                join_edges_set.add(edge)
                joins.append(edge)
            cur = nxt

    # Ensure joins are ordered so that left side is already joinable.
    # Simple iterative ordering:
    ordered_joins: List[Tuple[str, str, str, str]] = []
    remaining = joins[:]
    progress = True
    while remaining and progress:
        progress = False
        for edge in remaining[:]:
            left, right, left_col, right_col = edge
            if left in joined:
                ordered_joins.append(edge)
                joined.add(right)
                remaining.remove(edge)
                progress = True
    if remaining:
        # graph had edges but ordering failed (should be rare)
        # fallback: append remaining
        ordered_joins.extend(remaining)

    # assign aliases for all tables used in joins (including join-only tables)
    alias_map: Dict[str, str] = {}
    alias_counter = 1
    tables_in_joins: List[str] = [main_table]
    for left, right, _, _ in ordered_joins:
        if left not in tables_in_joins:
            tables_in_joins.append(left)
        if right not in tables_in_joins:
            tables_in_joins.append(right)
    for t in tables_to_select:
        if t not in tables_in_joins:
            tables_in_joins.append(t)
    for t in tables_in_joins:
        if t not in alias_map:
            alias_map[t] = f"T{alias_counter}"
            alias_counter += 1

    def qname(t: str) -> str:
        if schema_name:
            return f'{schema_name}.{t}'
        return t

    # SELECT clause
    select_parts: List[str] = []

    # selected_schema tables only
    for t, v in selected_schema.items():
        if isinstance(v, str) and v.lower() == "keep_all":
            select_parts.append(f'    {alias_map[t]}.*')
        elif isinstance(v, list):
            for col in v:
                select_parts.append(f'    {alias_map[t]}."{col}" AS {t}_{col}')

    if not select_parts:
        raise ValueError("No selectable columns after filtering ID/UUID types")

    # FROM + JOIN clauses
    sql_lines: List[str] = []
    sql_lines.append("SELECT")
    sql_lines.append(",\n".join(select_parts))
    sql_lines.append(f'FROM {qname(main_table)} AS {alias_map[main_table]}')

    for left, right, left_col, right_col in ordered_joins:
        sql_lines.append(
            f'LEFT JOIN {qname(right)} AS {alias_map[right]} '
            f'ON {alias_map[left]}."{left_col}" = {alias_map[right]}."{right_col}"'
        )

    return "```sql\n" + "\n".join(sql_lines) + "\n```"

def decomposer_node(state, llm: BaseChatModel):
    # schema_info = state['desc_str']
    fk_info = state['fk_str']
    query = state['query']
    selected_info = state.get("extracted_schema") or {}
    table_columns = state.get("table_columns", {}) or {}
    table_column_types = state.get("table_column_types", {}) or {}
    evidence = state.get('evidence')
    mode = state.get('analysis_mode','full_pipeline')

    if mode == 'data_exploration':

        prompt = decompose_template.format(
            fk_str=fk_info, query=query, evidence=evidence
        )

        llm_reply = call_llm(prompt, llm=llm)

        all_sqls = []
        for match in re.finditer(r'```sql(.*?)```', llm_reply, re.DOTALL):
            all_sqls.append(match.group(1).strip())
        if all_sqls:
            sql = all_sqls[-1]
        else:
            raise ValueError("No SQL found in the LLM response")

        return {
            **state,
            'final_sql': sql,
            'qa_pairs': llm_reply,
            'send_to': 'refiner_node',
            'messages': state['messages'] + [
                {"role": "decomposer", "content": llm_reply}
            ]
        }
    elif mode == "full_pipeline":
        # Expand any keep_all directives into explicit column lists so the LLM
        # can select the full set of variables from Treatment/Outcome tables.
        # table_columns = state.get('table_columns', {}) or {}
        # expanded_selected_info = {}
        # if isinstance(selected_info, dict):
        #     for tbl, decision in selected_info.items():
        #         if decision in ('keep_all', '', None):
        #             expanded_selected_info[tbl] = table_columns.get(tbl, [])
        #         elif isinstance(decision, list):
        #             expanded_selected_info[tbl] = decision
        #         else:
        #             # Unknown directive: keep as-is
        #             expanded_selected_info[tbl] = decision
        # else:
        #     expanded_selected_info = selected_info

        #     prompt = decompose_template_for_causal.format(
        #         fk_str=fk_info, query=query, selected_info=selected_info, evidence=evidence
        #     )
        # llm_reply = call_llm(prompt, llm=llm)

        if isinstance(selected_info, dict) and "selected_schema" in selected_info:
            selected_schema = selected_info.get("selected_schema", {})
        else:
            selected_schema = selected_info

        def _is_id_type(table: str, column: str) -> bool:
            col_type = table_column_types.get(table, {}).get(column)
            if col_type is None:
                return False
            type_str = str(col_type).lower()
            return type_str == "id" or "uuid" in type_str or "string" in type_str

        def _filter_id_columns(table: str, columns: List[str]) -> List[str]:
            return [col for col in columns if not _is_id_type(table, col)]

        expanded_selected_schema: Dict[str, object] = {}
        for table, decision in (selected_schema or {}).items():
            if isinstance(decision, str) and decision.lower() == "keep_all":
                cols = table_columns.get(table)
                if cols:
                    filtered = _filter_id_columns(table, cols)
                    expanded_selected_schema[table] = filtered
                else:
                    expanded_selected_schema[table] = decision
            elif isinstance(decision, list):
                expanded_selected_schema[table] = _filter_id_columns(table, decision)
            else:
                expanded_selected_schema[table] = decision

        if not isinstance(expanded_selected_schema, dict) or not expanded_selected_schema:
            raise ValueError("No selected_schema found for SQL generation")

        def _extract_table(ref: object) -> Optional[str]:
            if isinstance(ref, dict):
                return ref.get("table")
            if isinstance(ref, str) and "." in ref:
                return ref.split(".", 1)[0]
            return None

        main_table = _extract_table(state.get("treatment")) or choose_main_table(expanded_selected_schema)

        sql_md = generate_flat_dataset_sql(expanded_selected_schema, fk_info, main_table=main_table)
        sql = sql_md.replace("```sql", "").replace("```", "").strip()


        return {
            **state,
            'final_sql': sql,
            'send_to': 'refiner_node',
        }
