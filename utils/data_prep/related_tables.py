import json
from typing import Dict, List
from datetime import datetime, timezone
from utils.database import Database
from utils.redis_client import redis_client

# 타입 분류
NUMERIC_TYPES = ['integer', 'numeric', 'real', 'double precision', 'smallint', 'bigint']
DATE_TYPES = ['date', 'timestamp', 'timestamp without time zone', 'timestamp with time zone']
TEXT_TYPES = ['character varying', 'varchar', 'text']

def schema_to_comparable_json(schema: dict) -> str:
    return json.dumps(schema, default=str, sort_keys=True)

# 컬럼 통계 수집 함수
def get_column_stats(cursor, table: str, col: str, dtype: str) -> dict:
    stats = {}
    try:
        if dtype in NUMERIC_TYPES:
            cursor.execute(f"""
                SELECT COUNT(*), COUNT("{col}"), COUNT(DISTINCT "{col}"),
                       MIN("{col}"), MAX("{col}"), AVG("{col}"), STDDEV_POP("{col}"),
                       PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY "{col}"),
                       PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY "{col}"),
                       PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY "{col}")
                FROM "{table}"
            """)
            result = cursor.fetchone()

            stats.update({
                "count": result[0],
                "nulls": result[0] - result[1],
                "distinct": result[2],
                "min": result[3],
                "max": result[4],
                "avg": round(result[5], 3) if result[5] else None,
                "stddev": round(result[6], 3) if result[6] else None,
                "median(q2)": result[7],
                "q1": result[8],
                "q3": result[9]
            })

            if result[8] is not None and result[9] is not None:
                iqr = result[9] - result[8]
                lower_bound = result[8] - 1.5 * iqr
                upper_bound = result[9] + 1.5 * iqr
                cursor.execute(f"""
                    SELECT COUNT(*) FROM "{table}"
                    WHERE "{col}" < {lower_bound} OR "{col}" > {upper_bound}
                """)
                outlier_count = cursor.fetchone()[0]
                stats["iqr_outlier_count"] = outlier_count

        elif dtype in DATE_TYPES:
            cursor.execute(f"""
                SELECT COUNT(*), COUNT("{col}"), COUNT(DISTINCT "{col}"),
                       MIN("{col}"), MAX("{col}"),
                       PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY EXTRACT(EPOCH FROM "{col}"))
                FROM "{table}"
            """)
            result = cursor.fetchone()
            stats.update({
                "count": result[0],
                "nulls": result[0] - result[1],
                "distinct": result[2],
                "min": result[3],
                "max": result[4],
                "median": datetime.fromtimestamp(float(result[5]), tz=timezone.utc).isoformat() if result[5] else None
            })

        elif dtype == 'boolean':
            cursor.execute(f"""
                SELECT COUNT(*), COUNT("{col}"), COUNT(DISTINCT "{col}"),
                       MIN(CASE WHEN "{col}" IS NOT NULL THEN CAST("{col}" AS INT) END),
                       MAX(CASE WHEN "{col}" IS NOT NULL THEN CAST("{col}" AS INT) END)
                FROM "{table}"
            """)
            result = cursor.fetchone()
            stats.update({
                "count": result[0],
                "nulls": result[0] - result[1],
                "distinct": result[2],
                "min": result[3],
                "max": result[4]
            })

        else:
            cursor.execute(f"""
                SELECT COUNT(*), COUNT("{col}"), COUNT(DISTINCT "{col}")
                FROM "{table}"
            """)
            result = cursor.fetchone()
            stats.update({
                "count": result[0],
                "nulls": result[0] - result[1],
                "distinct": result[2]
            })

        if dtype in TEXT_TYPES:
            cursor.execute(f"""
                SELECT "{col}", COUNT(*) as freq
                FROM "{table}"
                GROUP BY "{col}"
                ORDER BY freq DESC
                LIMIT 3
            """)
            top_vals = cursor.fetchall()
            # stats["top_values"] = {str(k): v for k, v in top_vals}
            stats["top_values"] = {
                "values": [{"value": str(k), "freq": v} for k, v in top_vals]
            }

        cursor.execute(f"""
            SELECT DISTINCT "{col}"
            FROM "{table}"
            WHERE "{col}" IS NOT NULL
            LIMIT 3
        """)
        stats["examples"] = [str(result[0]) for result in cursor.fetchall()]

    except Exception as e:
        raise e
    return stats


def extract_schema(db_id) -> dict:
    database = Database()
    conn = database.get_connection(db_id)
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = 'public' AND table_type = 'BASE TABLE';
        """)
        tables = [result[0] for result in cursor.fetchall()]
        schema_info = {}

        for table in tables:
            cursor.execute(f"""
                SELECT column_name, data_type, is_nullable, column_default
                FROM information_schema.columns
                WHERE table_name = '{table}';
            """)
            columns_raw = cursor.fetchall()
            columns = {}

            for col, dtype, nullable, default in columns_raw:
                col_info = {
                    "type": dtype,
                    "nullable": (nullable == 'YES'),
                    "default": default
                }
                col_info.update(get_column_stats(cursor, table, col, dtype))
                columns[col] = col_info

            cursor.execute(f"""
                SELECT a.attname FROM pg_index i
                JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey)
                WHERE i.indrelid = '{table}'::regclass AND i.indisprimary;
            """)
            pk = [r[0] for r in cursor.fetchall()]
            for col in pk:
                if col in columns:
                    columns[col]["pk"] = True

            cursor.execute(f"""
                SELECT a.attname FROM pg_constraint c
                JOIN pg_class t ON c.conrelid = t.oid
                JOIN pg_attribute a ON a.attrelid = t.oid AND a.attnum = ANY(c.conkey)
                WHERE c.contype = 'u' AND t.relname = '{table}';
            """)
            uq = [r[0] for r in cursor.fetchall()]
            for col in uq:
                if col in columns:
                    columns[col]["unique"] = True

            cursor.execute(f"""
                SELECT kcu.column_name, ccu.table_name, ccu.column_name
                FROM information_schema.table_constraints AS tc
                JOIN information_schema.key_column_usage AS kcu ON tc.constraint_name = kcu.constraint_name
                JOIN information_schema.constraint_column_usage AS ccu ON ccu.constraint_name = tc.constraint_name
                WHERE tc.constraint_type = 'FOREIGN KEY' AND tc.table_name = '{table}';
            """)
            fk_tuples = cursor.fetchall()
            foreign_keys = []
            for col, ref_table, ref_col in fk_tuples:
                foreign_keys.append((col, ref_table, ref_col))
                if col in columns:
                    columns[col]["fk"] = f"{ref_table}.{ref_col}"

            cursor.execute(f"""
                SELECT conname, pg_get_expr(conbin, conrelid)
                FROM pg_constraint
                WHERE contype = 'c' AND conrelid = '{table}'::regclass;
            """)
            checks = cursor.fetchall()
            check_constraints = [expr for _, expr in checks]

            schema_info[table] = {
                "columns": columns,
                "primary_key": pk,
                "foreign_keys": foreign_keys,
                "check_constraints": check_constraints
            }
        return schema_info
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        cursor.close()
        conn.close()

def generate_edges(schema_info: dict) -> Dict[str, List[str]]:
    edges = {}
    for table, info in schema_info.items():
        for _, ref_table, _ in info.get("foreign_keys", []):
            edges.setdefault(table, []).append(ref_table)
    return edges

def generate_edge_reason(edges: Dict[str, List[str]], schema_info: dict) -> Dict[str, List[dict]]:
    metadata = {}
    for from_table, to_tables in edges.items():
        for to_table in to_tables:
            fk_info = schema_info[from_table].get("foreign_keys", [])
            matched = [fk for fk in fk_info if fk[1] == to_table]
            key = f"{from_table}→{to_table}"
            metadata[key] = []

            for fk_col, ref_table, ref_col in matched:
                col_info = schema_info[from_table]["columns"].get(fk_col, {})
                nullable = col_info.get("nullable", True)
                unique = col_info.get("unique", False)
                pk = col_info.get("pk", False)

                modality = "mandatory" if not nullable else "optional"
                cardinality = "1:1" if unique or pk else "1:N"

                metadata[key].append({
                    "from_column": fk_col,
                    "to_column": ref_col,
                    "cardinality": cardinality,
                    "modality": modality,
                    "reason": f"Linked via {from_table}.{fk_col} → {to_table}.{ref_col}, {cardinality}, {modality} relationship",
                    "source": "auto"
                })
    return metadata

def update_table_relations(db_id) -> bool:
    new_schema = extract_schema(db_id)
    stored = redis_client.get(f"{db_id}:table_relations")
    stored_schema = json.loads(stored)["source_schema"] if stored else {}
    
    if schema_to_comparable_json(new_schema) != schema_to_comparable_json(stored_schema):
        print("Schema change detected. Updating Redis...")
        edges = generate_edges(new_schema)
        edge_reasons = generate_edge_reason(edges, new_schema)

        redis_client.set(f"{db_id}:table_relations", json.dumps({
            "edges": edges,
            "edge_reasons": edge_reasons,
            "source_schema": new_schema
        }, indent=2, default=str))
        print("Schema updated in Redis.")
        return new_schema, True
    
    # print("Did not update because schema did not change.") # for debugging
    return new_schema, False