import json
import datetime
import graphviz
from utils.redis_client import redis_client

def generate_erd(state):
    db_id = state.get("db_id")
    recommended = state["recommended_tables"]
    metadata = {}
    for table in recommended:
        redis_key = f"{db_id}:metadata:{table}"
        try:
            raw = redis_client.get(redis_key)
            if raw:
                metadata[table] = json.loads(raw)
        except Exception as e:
            print(f"ERROR: Failed to load metadata for {table}: {e}")

    erd = graphviz.Digraph(format='png')
    erd.attr('node', shape='plaintext')

    for table, data in metadata.items():
        rows = [f'<tr><td bgcolor="lightblue" colspan="2"><b>{table}</b></td></tr>']
        for col, info in data["schema"]["columns"].items():
            dtype = info["type"]
            extras = []
            if info.get("pk"): extras.append("PK")
            if info.get("unique"): extras.append("UQ")
            if info.get("fk"): extras.append("FK")
            row = f"<tr><td>{col}</td><td>{dtype} {' '.join(extras)}</td></tr>"
            rows.append(row)
        label = f"""<
        <table border="1" cellborder="1" cellspacing="0">
        {''.join(rows)}
        </table>
        >"""
        erd.node(table, label=label)

    for src_table, data in metadata.items():
        for col, info in data["schema"]["columns"].items():
            if "fk" in info:
                fk_target = info["fk"]
                ref_table, ref_col = fk_target.split(".")
                modality = "odot" if info.get("nullable", True) else "tee"
                cardinality = "crow" if not info.get("unique", False) and not info.get("pk", False) else "none"
                erd.edge(
                    ref_table,
                    src_table,
                    arrowtail=modality,
                    arrowhead=cardinality,
                    dir="both"
                )

    output_path = f"./outputs/images/erd/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    erd.render(output_path, cleanup=True)

    # final_output
    final_output = {
    "objective_summary": state.get("objective_summary", "No summary found. Something went wrong!"),
    "recommended_tables": state.get("recommended_tables", []),
    "recommended_method": state.get("recommended_method", "No method recommended. Something went wrong!"),
    "erd_image_path": output_path + ".png",
    }

    return {
        "erd_image_path": output_path + ".png",
        "final_output": final_output
    }