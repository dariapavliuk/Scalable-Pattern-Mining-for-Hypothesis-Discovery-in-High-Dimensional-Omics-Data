# Example config for run_syflow_pipeline.py
# Copy this file to configs/<your_exp>.py and fill in your values.

# =========================
# Relevance
# =========================

# Path to CSV with per-feature relevance scores (output of limma or similar)
RELEVANCE_PATH = "/path/to/your/relevance.csv"

# Column name in that CSV to use as the relevance score (e.g. "t", "logFC")
RELEVACE_COL = "t"

# =========================
# SyFlow
# =========================

# Number of subgroups to discover
N_SUBGROUPS = 2

# Grid of SyFlow hyperparameters to sweep over
HYPER_PARAMS = {
    "temperature": [0.1, 0.2, 0.3],        
    "alpha": [0.2, 0.3, 0.4],              
    "lambd": [2],                           
    "pop_train_epochs": [1000],            
    "subgroup_train_epochs": [1000],        
    "lr_flow": [0.05],                    
    "lr_classifier": [0.02],               
}

# Maximum percentage of samples for which the predicate is TRUE
THRESHOLD = 0.75

# =========================
# Dask cluster
# =========================

# Number of parallel workers
N_WORKERS = 8
THREADS_PER_WORKER = 1

# RAM limit per worker (passed to Dask LocalCluster)
MEMORY_LIMIT = "16GB"

# Timeout per individual run in seconds (86400 = 24 hours)
TIMEOUT_SEC = 86400

# =========================
# Pipeline
# =========================

# Number of target variables to try in a batch
BATCH_SIZE = 10

# Maximum number of batches
MAX_ITERATIONS = 5

# Desired number of rules in the rule set
N = 20

# Unique experiment name (used to name output files / log entries)
EXP_NAME = "my_experiment"

# Path to the input data CSV (samples x features, with index column)
DATA_PATH = "/path/to/your/data.csv"

# Candidate values for the minimum subgroup size (fraction of total samples)
MIN_SUBGROUP_SIZES = [0.05, 0.1, 0.15, 0.2]

# Candidate values for the minimum KL divergence between subgroups
MIN_KLS = [0.5, 0.6, 0.7, 0.8]

# Whether to use redundancy penalisation in mRMR feature selection
USE_REDUNDANCY = True

# =========================
# Log schema
# =========================
# These should not need to change unless you modify run_and_log_syflow().

COLUMNS_LOG_SYFLOW = [
    "run_id",
    "local_exp_name",
    "target_col",
    "temperature",
    "alpha",
    "lambd",
    "pop_train_epochs",
    "subgroup_train_epochs",
    "lr_flow",
    "lr_classifier",
    "start_time",
    "end_time",
    "duration_sec",
    "success",
    "status",
    "error_message",
    "result_path",
]

DTYPES_LOG_SYFLOW = {
    "run_id": "string",
    "local_exp_name": "string",
    "target_col": "string",
    "temperature": "float64",
    "alpha": "float64",
    "lambd": "float64",
    "pop_train_epochs": "int64",
    "subgroup_train_epochs": "int64",
    "lr_flow": "float64",
    "lr_classifier": "float64",
    "start_time": "string",
    "end_time": "string",
    "duration_sec": "float64",
    "success": "boolean",
    "status": "string",
    "error_message": "string",
    "result_path": "string",
}
