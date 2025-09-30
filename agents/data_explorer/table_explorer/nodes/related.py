import json
from utils.redis_client import redis_client
from utils.data_prep.related_tables import update_table_relations

# -------------------------------
# related_tables_node
# -------------------------------
def get_related_graph(graph: dict, table_name: str) -> dict:
    related = {}
    # outbound
    for to_table in graph["edges"].get(table_name, []):
        key = f"{table_name}→{to_table}"
        edge_data = graph.get("edge_reasons", {}).get(key, [])
        reasons = [d.get("reason", "No reason found") for d in edge_data]
        related[to_table] = "\n".join(reasons) if reasons else "No reason found"
    # inbound
    for from_table, to_tables in graph["edges"].items():
        if table_name in to_tables:
            key = f"{from_table}→{table_name}"
            edge_data = graph.get("edge_reasons", {}).get(key, [])
            reasons = [d.get("reason", "No reason found") for d in edge_data]
            related[from_table] = "\n".join(reasons) if reasons else "No reason found"
    return related

def related_tables(table_name: str, db_id: str) -> str:
    update_table_relations(db_id)
    graph_raw = redis_client.get(f"{db_id}:table_relations")
    graph = json.loads(graph_raw) if graph_raw else {}
    related = get_related_graph(graph, table_name)
    return related

def related_tables_node(state):
    table_name = state["input"]
    related = related_tables(table_name, state["db_id"])
    return {
        "related_tables": related
    }