import importlib.util
import os
import pickle
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from dask.distributed import Client, LocalCluster

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.apply_syflow_omicarules import (
    build_combinations,
    load_done_keys,
    ensure_log_csv,
    run_and_log_syflow,
    exp_key_syflow,
)
from src.process_syflow_results import (
    process_syflow_results_wrapper,
    select_best_combo,
)

# =========================
# CONFIG
# =========================

CONFIG_FILE = "configs/config_example_mrna.py"  # <-- set your config here


def load_config(path: str):
    spec = importlib.util.spec_from_file_location("config", Path(__file__).resolve().parent / path)
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)
    return cfg


cfg = load_config(CONFIG_FILE)


# =========================
# PIPELINE
# =========================

def mrmr_greedy(
    data_df,
    relevance_dict,
    redundancy_matrix=None,
    k=None,
    initial_selected=None,
    use_redundancy=True,
):
    features = list(data_df.columns)
    relevance = pd.Series(relevance_dict).reindex(features)

    if initial_selected is None:
        selected = []
    else:
        selected = list(initial_selected)

    if k is None:
        k = len(features) - len(selected)

    remaining = set(features) - set(selected)

    for step in range(k):
        best_feat = None
        best_score = -np.inf

        for feat in remaining:
            rel = relevance[feat]

            if use_redundancy and selected and redundancy_matrix is not None:
                redund = redundancy_matrix.loc[feat, selected].mean()
            else:
                redund = 0.0

            score = rel - redund

            if score > best_score:
                best_score = score
                best_feat = feat

        selected.append(best_feat)
        remaining.remove(best_feat)

    return selected


def exp_key_pipeline_syflow(n, min_subgroup_size, min_kl, use_redundancy):
    return (
        f"N{n}"
        f"_n{min_subgroup_size}"
        f"_kl{min_kl}"
        f"_redundancy{use_redundancy}"
    )


def _build_output_paths(exp_name):
    output_dir = str(_PROJECT_ROOT / "results" / "run_syflow" / exp_name)
    return {
        "output_dir": output_dir,
        "results_dir": f"{output_dir}/results",
        "pipeline_results_dir": f"{output_dir}/pipeline_results",
        "log_file": os.path.join(output_dir, "log.csv"),
    }


def _load_data(data_path, relevance_path, relevance_col):
    data_df = pd.read_csv(data_path, index_col=0)
    relevance_df = pd.read_csv(relevance_path, index_col=0)
    raw = relevance_df[relevance_col].abs()
    raw = (raw - raw.min()) / (raw.max() - raw.min())
    relevance_dict = raw.to_dict()
    redundancy_matrix = data_df.corr(method="pearson").abs()
    return data_df, relevance_dict, redundancy_matrix


def _make_dask_client(n_workers, threads_per_worker, memory_limit):
    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        memory_limit=memory_limit,
        processes=True,
        dashboard_address=None,
    )
    client = Client(cluster)
    print("Scheduler:", cluster.scheduler_address)
    print("Cluster:", cluster)
    return client, cluster


def _evaluate_combo(
    combo, log_df, data_df, target_col, relevance_dict, threshold, min_kl, min_subgroup_size
):
    key = exp_key_syflow(combo)

    row = log_df[
        (log_df["local_exp_name"] == key)
        & (log_df["success"] == True)
    ]
    if row.empty:
        return None

    path = row["result_path"].values[0]
    with open(path, "rb") as f:
        syflow_results = pickle.load(f)

    processed = process_syflow_results_wrapper(
        syflow_results["rules"],
        data_df,
        target_col,
        relevance_dict,
        threshold,
    )

    if processed is None:
        print("Processed is None for target:", target_col)
        return None

    metrics = processed[0]

    if metrics is None:
        print(f"Metrics is None for target {target_col}")
        return None

    if "kl_divergence" not in metrics or "proportional_subgroup_size" not in metrics:
        print(f"Missing keys in metrics for target {target_col}")
        return None

    passes_final = (
        metrics["kl_divergence"] >= min_kl
        and metrics["proportional_subgroup_size"] >= min_subgroup_size
    )
    if not passes_final:
        return None

    return key, combo, processed


def run_pipeline(n, min_subgroup_size, min_kl, use_redundancy):
    paths = _build_output_paths(cfg.EXP_NAME)
    os.makedirs(paths["results_dir"], exist_ok=True)
    os.makedirs(paths["pipeline_results_dir"], exist_ok=True)
    ensure_log_csv(paths["log_file"], cfg.COLUMNS_LOG_SYFLOW, cfg.DTYPES_LOG_SYFLOW)

    data_df, relevance_dict, redundancy_matrix = _load_data(
        cfg.DATA_PATH, cfg.RELEVANCE_PATH, cfg.RELEVACE_COL
    )

    client, cluster = _make_dask_client(cfg.N_WORKERS, cfg.THREADS_PER_WORKER, cfg.MEMORY_LIMIT)

    mrmr_ranking = mrmr_greedy(
        data_df,
        relevance_dict,
        redundancy_matrix,
        k=len(data_df.columns),
        use_redundancy=use_redundancy,
    )

    results_summary = {}
    tested_targets = []
    num_success = 0
    iteration = 0
    cursor = 0

    while (
        num_success < n
        and iteration < cfg.MAX_ITERATIONS
        and cursor < len(mrmr_ranking)
    ):
        iteration += 1

        batch = mrmr_ranking[cursor : cursor + cfg.BATCH_SIZE]
        batch = [t for t in batch if t not in tested_targets]

        if not batch:
            break

        cursor += len(batch)

        params = {"target_col": batch, **cfg.HYPER_PARAMS}
        done = load_done_keys(paths["log_file"], cfg.DTYPES_LOG_SYFLOW)
        all_combos = list(build_combinations(params))
        combos = [c for c in all_combos if exp_key_syflow(c) not in done]

        futures = [
            client.submit(
                run_and_log_syflow,
                data_df,
                combo,
                cfg.N_SUBGROUPS,
                paths["results_dir"],
                paths["log_file"],
                cfg.COLUMNS_LOG_SYFLOW,
                cfg.DTYPES_LOG_SYFLOW,
                pure=False,
            )
            for combo in combos
        ]
        client.gather(futures)
        tested_targets.extend(batch)

        log_df = pd.read_csv(paths["log_file"])

        for target_col in batch:
            combos_t = [c for c in all_combos if c["target_col"] == target_col]

            valid_results = {}
            for combo in combos_t:
                result = _evaluate_combo(
                    combo, log_df, data_df, target_col,
                    relevance_dict, cfg.THRESHOLD, min_kl, min_subgroup_size,
                )
                if result is not None:
                    key, combo_out, processed = result
                    valid_results[key] = (combo_out, processed)

            if not valid_results:
                results_summary[target_col] = {"success": False}
                continue

            best_key, best_result = select_best_combo(valid_results)
            results_summary[target_col] = {"best_result": best_result, "success": True}
            num_success += 1

            if num_success >= n:
                break

    results_summary_path = os.path.join(
        paths["pipeline_results_dir"],
        f"{exp_key_pipeline_syflow(n, min_subgroup_size, min_kl, use_redundancy)}.pkl",
    )
    with open(results_summary_path, "wb") as f:
        pickle.dump(results_summary, f)

    client.close()
    cluster.close()


if __name__ == "__main__":
    for min_subgroup_size in cfg.MIN_SUBGROUP_SIZES:
        for min_kl in cfg.MIN_KLS:
            run_pipeline(n=cfg.N, min_subgroup_size=min_subgroup_size, min_kl=min_kl, use_redundancy=cfg.USE_REDUNDANCY)
