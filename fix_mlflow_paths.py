"""
Helper script to update artifact paths in mlflow.db for local/cross-platform environments (e.g. Mac / Linux).

Usage:
    python fix_mlflow_paths.py
"""

import os
import sqlite3


def fix_paths(db_path="mlflow.db"):
    if not os.path.exists(db_path):
        print(f"Error: {db_path} not found.")
        return

    repo_dir = os.path.abspath(".").replace("\\", "/")
    mlruns_dir = f"{repo_dir}/mlruns"

    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # Update experiments
    c.execute(
        "UPDATE experiments SET artifact_location = ? WHERE experiment_id = '1'",
        (f"file:{mlruns_dir}/1",),
    )
    c.execute(
        "UPDATE experiments SET artifact_location = ? WHERE experiment_id = '0'",
        (f"file:{mlruns_dir}/0",),
    )

    # Update runs
    rows = c.execute("SELECT run_uuid FROM runs").fetchall()
    for (run_uuid,) in rows:
        new_uri = f"file:{mlruns_dir}/1/{run_uuid}/artifacts"
        c.execute("UPDATE runs SET artifact_uri = ? WHERE run_uuid = ?", (new_uri, run_uuid))

    conn.commit()
    conn.close()
    print(f"Successfully updated mlflow.db artifact paths to:\n  {mlruns_dir}")


if __name__ == "__main__":
    fix_paths()
