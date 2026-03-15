import importlib.util
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from dask.distributed import Client, LocalCluster
from src.apply_syflow_omicarules import (
    build_combinations,
    load_done_keys,
    ensure_log_csv,
    run_and_log_omicsarules,
    exp_key_omicsarules,
)

# =========================
# CONFIG
# =========================

CONFIG_FILE = "configs/config_biobank.py"  # <-- set your config here


def load_config(path: str):
    spec = importlib.util.spec_from_file_location("config", Path(__file__).resolve().parent / path)
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)
    return cfg


cfg = load_config(CONFIG_FILE)

R_SCRIPT_PATH = str(Path(__file__).resolve().parent / "r_script_omicsarules.R")
OMICSARULES_DIR = str(_PROJECT_ROOT / "omicsArules")

OUTPUT_DIR = str(_PROJECT_ROOT / "results" / "run_omicsarules" / cfg.EXP_NAME)
RESULTS_DIR = f"{OUTPUT_DIR}/results"
LOGS_DIR = f"{OUTPUT_DIR}/logs"
LOG_FILE = os.path.join(OUTPUT_DIR, "log.csv")

COLUMNS_LOG_OMICSARULES = [
    "run_id", "local_exp_name", "supp", "conf", "maxl", "minl",
    "start_time", "end_time", "duration_sec", "success", "status",
    "error_message", "result_path",
]

DTYPES_LOG_OMICSARULES = {
    "run_id": "string", "local_exp_name": "string", "supp": "float64",
    "conf": "float64", "maxl": "int64", "minl": "int64",
    "start_time": "string", "end_time": "string", "duration_sec": "float64",
    "success": "boolean", "status": "string",
    "error_message": "string", "result_path": "string",
}


# =========================
# MAIN
# =========================

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    ensure_log_csv(LOG_FILE, COLUMNS_LOG_OMICSARULES, DTYPES_LOG_OMICSARULES)

    done = load_done_keys(LOG_FILE, DTYPES_LOG_OMICSARULES)

    combos = [
        c for c in build_combinations(cfg.PARAMS)
        if exp_key_omicsarules(c) not in done
    ]

    dask_cluster = LocalCluster(
        n_workers=cfg.N_WORKERS,
        threads_per_worker=cfg.THREADS_PER_WORKER,
        processes=True,
        dashboard_address=None,
        memory_limit=0,  # Dask cannot see R subprocesses; memory managed via psutil
    )
    client = Client(dask_cluster)

    print("Scheduler:", dask_cluster.scheduler_address)
    print("Dashboard:", client.dashboard_link)
    print("Cluster:", dask_cluster)

    futures = [
        client.submit(
            run_and_log_omicsarules,
            R_SCRIPT_PATH,
            cfg.DATA_PATH,
            combo,
            RESULTS_DIR,
            LOGS_DIR,
            LOG_FILE,
            COLUMNS_LOG_OMICSARULES,
            DTYPES_LOG_OMICSARULES,
            cfg.TIMEOUT_SEC,
            cfg.R_MEMORY_LIMIT_GB,
            OMICSARULES_DIR,
            pure=False,
        )
        for combo in combos
    ]

    client.gather(futures)
    client.close()
    dask_cluster.close()


if __name__ == "__main__":
    main()
