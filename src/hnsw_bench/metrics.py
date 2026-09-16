"""Postgres-side size and dead-tuple metrics for the active table.
n_dead_tup is an estimate refreshed periodically by the stats
subsystem, not an exact count - see PLAN.md Sec 1-2 item 8.
"""

import psycopg


def collect_metrics(cur: psycopg.Cursor, table: str = "active") -> dict:
    cur.execute("SELECT pg_table_size(%s), pg_indexes_size(%s)", (table, table))

    row = cur.fetchone()
    if row is None:
        raise RuntimeError("Expected table/index size result")
    table_bytes, index_bytes = row

    cur.execute(
        "SELECT n_dead_tup FROM pg_stat_user_tables WHERE relname = %s", (table,)
    )
    row = cur.fetchone()
    return {
        "table_size_bytes": table_bytes,
        "index_size_bytes": index_bytes,
        "dead_tuples": row[0] if row else 0,
    }
