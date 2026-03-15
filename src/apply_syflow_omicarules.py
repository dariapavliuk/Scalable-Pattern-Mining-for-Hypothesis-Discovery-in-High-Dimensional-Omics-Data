import subprocess
import pandas as pd
import os
import time
import uuid
from datetime import datetime
import itertools
import pickle
import threading
import psutil
import sys
from pathlib import Path

import flowtorch.bijectors as bij

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from syflow.src.methods import run_syflow

TERMINAL = {"finished", "timeout", "memory_exceeded", "r_error"}

_STATUS_MAP = {
    0:   ("finished",        True),
    124: ("timeout",         False),
    137: ("memory_exceeded", False),
}

_ERROR_MSG = {
    124: "timeout",
    137: "memory limit exceeded",
}


def exp_key_omicsarules(combo):
    return (
        f"supp{combo['supp']}"
        f"_conf{combo['conf']}"
        f"_maxl{combo['maxl']}"
        f"_minl{combo['minl']}"
    )


def exp_key_syflow(combo):
    return (
        f"{combo['target_col']}"
        f"_temp{combo['temperature']}"
        f"_alpha{combo['alpha']}"
        f"_lambd{combo['lambd']}"
        f"_pe{combo['pop_train_epochs']}"
        f"_se{combo['subgroup_train_epochs']}"
        f"_lrf{combo['lr_flow']}"
        f"_lrc{combo['lr_classifier']}"
    )


def ensure_log_csv(log_path, columns, dtypes):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    if not os.path.exists(log_path):
        df = pd.DataFrame({c: pd.Series(dtype=dtypes[c]) for c in columns})
        df.to_csv(log_path, index=False)


def _with_file_lock(lock_path):
    import fcntl

    class _LockCtx:
        def __enter__(self):
            self.f = open(lock_path, "a+")
            fcntl.flock(self.f.fileno(), fcntl.LOCK_EX)
            return self.f

        def __exit__(self, exc_type, exc, tb):
            self.f.flush()
            os.fsync(self.f.fileno())
            fcntl.flock(self.f.fileno(), fcntl.LOCK_UN)
            self.f.close()

    return _LockCtx()


def _atomic_write_csv(df, path):
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def safe_append_running_row(log_path, columns, dtypes, row):
    lock_path = log_path + ".lock"
    ensure_log_csv(log_path, columns, dtypes)

    with _with_file_lock(lock_path):
        df = pd.read_csv(log_path, dtype=dtypes)
        new_row = pd.DataFrame([row]).astype(dtypes)
        df = pd.concat([df, new_row], ignore_index=True)
        _atomic_write_csv(df, log_path)


def safe_update_row_by_run_id(log_path, run_id, dtypes, updates):
    lock_path = log_path + ".lock"

    with _with_file_lock(lock_path):
        df = pd.read_csv(log_path, dtype=dtypes)

        idx = df.index[df["run_id"].astype(str) == str(run_id)].tolist()
        if not idx:
            raise RuntimeError(f"run_id {run_id} not found in log")

        i = idx[-1]

        for k, v in updates.items():
            df.loc[i, k] = v

        _atomic_write_csv(df, log_path)


def load_done_keys(log_path, dtypes):
    if not os.path.exists(log_path):
        return set()

    df = pd.read_csv(log_path, dtype=dtypes)

    return set(
        df.loc[df["status"].isin(TERMINAL), "local_exp_name"].astype(str)
    )


def build_combinations(params):
    keys = list(params.keys())
    for values in itertools.product(*params.values()):
        yield dict(zip(keys, values))


def _monitor_memory(proc, max_bytes, killed_flag, stop_event):
    """Poll RSS of R process + children every second; kill if over limit."""
    while not stop_event.is_set():
        try:
            p = psutil.Process(proc.pid)
            rss = p.memory_info().rss
            rss += sum(c.memory_info().rss for c in p.children(recursive=True))
            if rss > max_bytes:
                p.kill()
                killed_flag.append(True)
                return
        except psutil.NoSuchProcess:
            return
        stop_event.wait(1.0)


def apply_omicsarules(
    r_script_path,
    data_path,
    combo,
    out_path,
    timeout_sec,
    memory_limit_gb,
    omicsarules_dir=None,
):
    cmd = [
        "Rscript",
        r_script_path,
        f"data_path={data_path}",
        f"supp={combo['supp']}",
        f"conf={combo['conf']}",
        f"maxl={combo['maxl']}",
        f"minl={combo['minl']}",
        f"out_path={out_path}",
    ]
    if omicsarules_dir is not None:
        cmd.append(f"omicsarules_dir={omicsarules_dir}")

    max_bytes = int(memory_limit_gb * 1024**3)
    killed_flag = []
    stop_event = threading.Event()

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        monitor = threading.Thread(
            target=_monitor_memory,
            args=(proc, max_bytes, killed_flag, stop_event),
            daemon=True,
        )
        monitor.start()

        try:
            stdout, stderr = proc.communicate(timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate()
            stop_event.set()
            monitor.join(timeout=2.0)
            return subprocess.CompletedProcess(
                cmd,
                returncode=124,
                stdout=stdout or "",
                stderr=(stderr or "") + "\nTIMEOUT",
            )
        finally:
            stop_event.set()
            monitor.join(timeout=2.0)

        if killed_flag or proc.returncode < 0:
            return subprocess.CompletedProcess(
                cmd,
                returncode=137,
                stdout=stdout or "",
                stderr=(stderr or "") + "\nMEMORY LIMIT EXCEEDED",
            )

        return subprocess.CompletedProcess(
            cmd, returncode=proc.returncode, stdout=stdout, stderr=stderr
        )

    except MemoryError:
        return subprocess.CompletedProcess(
            cmd,
            returncode=137,
            stdout="",
            stderr="MEMORY LIMIT EXCEEDED",
        )


def run_and_log_omicsarules(
    r_script_path,
    data_path,
    combo,
    results_dir,
    logs_dir,
    log_path,
    columns,
    dtypes,
    timeout_sec,
    memory_limit_gb,
    omicsarules_dir=None,
):
    run_id = str(uuid.uuid4())
    local_exp_name = exp_key_omicsarules(combo)

    start_time_epoch = time.time()
    start_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    out_path = os.path.join(results_dir, f"{local_exp_name}.csv")
    pkl_path = os.path.join(logs_dir, f"{local_exp_name}.pkl")

    safe_append_running_row(
        log_path,
        columns,
        dtypes,
        {
            "run_id": run_id,
            "local_exp_name": local_exp_name,
            "supp": float(combo["supp"]),
            "conf": float(combo["conf"]),
            "maxl": int(combo["maxl"]),
            "minl": int(combo["minl"]),
            "start_time": start_ts,
            "end_time": None,
            "duration_sec": None,
            "success": False,
            "status": "running",
            "error_message": None,
            "result_path": None,
        },
    )

    result = None

    try:
        result = apply_omicsarules(
            r_script_path,
            data_path,
            combo,
            out_path,
            timeout_sec,
            memory_limit_gb,
            omicsarules_dir=omicsarules_dir,
        )

        status, success = _STATUS_MAP.get(result.returncode, ("r_error", False))
        error_message = (
            None if success
            else _ERROR_MSG.get(result.returncode, f"r_error (returncode={result.returncode})")
        )

        updates = {
            "end_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "duration_sec": round(time.time() - start_time_epoch, 2),
            "success": success,
            "status": status,
            "error_message": error_message,
            "result_path": out_path if success else None,
        }

    except Exception as e:
        updates = {
            "end_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "duration_sec": round(time.time() - start_time_epoch, 2),
            "success": False,
            "status": "python_error",
            "error_message": str(e),
            "result_path": None,
        }

    finally:
        safe_update_row_by_run_id(log_path, run_id, dtypes, updates)

        with open(pkl_path, "wb") as f:
            pickle.dump(
                {
                    "stdout": None if result is None else result.stdout,
                    "stderr": None if result is None else result.stderr,
                    "returncode": None if result is None else result.returncode,
                },
                f,
            )


class SyFlow_Config:
    def __init__(self, alpha=0.3, lambd=2, temperature=0.2,
                 pop_train_epochs=1000, subgroup_train_epochs=1000,
                 lr_flow=5e-2, lr_classifier=2e-2):
        self.alpha = alpha
        self.lambd = lambd
        self.temperature = temperature
        self.pop_train_epochs = pop_train_epochs
        self.subgroup_train_epochs = subgroup_train_epochs
        self.lr_flow = lr_flow
        self.lr_classifier = lr_classifier
        self.final_fit_epochs = 0
        self.bin_deviation = 0.2
        self.use_weights = True
        self.seed = 10

        def flow_gen():
            return bij.Spline(count_bins=12)
        self.flow_gen = flow_gen


def apply_syflow(df, target_col, n_subgroups, **kwargs):
    config = SyFlow_Config(**kwargs)
    features = df.drop(columns=[target_col])
    target = df[target_col]
    feature_names = features.columns.to_list()
    X = features.values
    Y = target.values.reshape(-1, 1)
    subgroups, rules = run_syflow(X, Y, config, n_subgroups, feature_names)
    return subgroups, rules


def run_and_log_syflow(df, combo, n_subgroups, results_dir, log_path, columns, dtypes):
    run_id = str(uuid.uuid4())
    local_exp_name = exp_key_syflow(combo)

    start_time_epoch = time.time()
    start_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, f"{local_exp_name}.pkl")

    safe_append_running_row(log_path, columns, dtypes,
        {
            "run_id": run_id,
            "local_exp_name": local_exp_name,
            "target_col": combo["target_col"],
            "temperature": combo["temperature"],
            "alpha": combo["alpha"],
            "lambd": combo["lambd"],
            "pop_train_epochs": combo["pop_train_epochs"],
            "subgroup_train_epochs": combo["subgroup_train_epochs"],
            "lr_flow": combo["lr_flow"],
            "lr_classifier": combo["lr_classifier"],
            "start_time": start_ts,
            "end_time": None,
            "duration_sec": None,
            "success": False,
            "status": "running",
            "error_message": None,
            "result_path": None,
        },
    )

    try:
        subgroups, rules = apply_syflow(
            df, combo["target_col"], n_subgroups,
            temperature=combo["temperature"],
            alpha=combo["alpha"],
            lambd=combo["lambd"],
            pop_train_epochs=combo["pop_train_epochs"],
            subgroup_train_epochs=combo["subgroup_train_epochs"],
            lr_flow=combo["lr_flow"],
            lr_classifier=combo["lr_classifier"],
        )

        updates = {
            "end_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "duration_sec": round(time.time() - start_time_epoch, 2),
            "success": True,
            "status": "finished",
            "error_message": None,
            "result_path": out_path,
        }

        with open(out_path, "wb") as f:
            pickle.dump({
                "target_col": combo["target_col"],
                "temperature": combo["temperature"],
                "alpha": combo["alpha"],
                "lambd": combo["lambd"],
                "subgroups": subgroups,
                "rules": rules,
            }, f)

    except Exception as e:
        updates = {
            "end_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "duration_sec": round(time.time() - start_time_epoch, 2),
            "success": False,
            "status": "python_error",
            "error_message": str(e),
            "result_path": None,
        }

    finally:
        safe_update_row_by_run_id(log_path, run_id, dtypes, updates)
