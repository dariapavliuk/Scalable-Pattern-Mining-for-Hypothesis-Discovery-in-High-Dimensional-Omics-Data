# Example config for run_omicsarules.py
# Copy this file to configs/<your_exp>.py and fill in your values.

# Path to the input data CSV (samples x features, with index column)
DATA_PATH = "/path/to/your/data.csv"

# Unique name for this experiment (used to name the output directory)
EXP_NAME = "my_experiment"

# Grid of omicsARules hyperparameters to sweep over
PARAMS = {
    "supp": [0.4, 0.2, 0.1, 0.05],   # minimum support
    "conf": [0.8, 0.6, 0.4, 0.2],    # minimum confidence
    "maxl": [3, 5, 7, 9, 11],        # maximum rule length
    "minl": [2],                      # minimum rule length
}

# Number of parallel Dask workers (each spawns one R subprocess)
N_WORKERS = 8
THREADS_PER_WORKER = 1

# RAM limit per R subprocess in GB (monitored via psutil)
# Total RAM usage ~ N_WORKERS * R_MEMORY_LIMIT_GB
R_MEMORY_LIMIT_GB = 16

# Timeout per experiment run in seconds (86400 = 24 hours)
TIMEOUT_SEC = 86400
