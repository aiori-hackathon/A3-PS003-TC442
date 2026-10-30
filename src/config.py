"""
config.py - every setting of the project lives here.

Change values here instead of editing the code in other files. Each experiment
saves a copy of the settings it used, so results stay reproducible.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
MAT_FILE = DATA_RAW / "traffic_dataset.mat"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
REPORTS_DIR = ROOT / "reports"

DATASET_URL = "https://archive.ics.uci.edu/static/public/608/traffic+flow+forecasting.zip"

# ---------------------------------------------------------------------------
# Dataset layout - VERIFIED on the real file with step2b_diagnose_layout.py.
# The real file has 48 columns (the documentation says 47):
#   0-9   10 lags of traffic volume, oldest -> newest (column 9 = most recent)
#   10-14 day indicator, one-hot with 5 categories (documentation says 7)
#   15-38 hour of day, one-hot (24)
#   39    number of lanes (1-5)
#   40-43 road direction, one-hot (4)
#   44-47 road name, one-hot (4)
# Column ranges are (start, end) with end excluded.
# ---------------------------------------------------------------------------
COLS_LAGS = (0, 10)
COLS_DAY = (10, 15)
COLS_HOUR = (15, 39)
COL_LANES = 39
COLS_DIRECTION = (40, 44)
COLS_ROAD = (44, 48)
NUM_RAW_COLUMNS = 48

# Prediction target.
# In the file, the target tra_Y is 2 steps (30 min) after the newest lag of the
# same row. Rows are consecutive, so the true next-15-minute value can be read
# from the next row's newest lag.
#   "next_15min" : predict the value 15 minutes after the newest reading (default)
#   "dataset"    : use tra_Y exactly as provided
TARGET_MODE = "next_15min"

# ---------------------------------------------------------------------------
# Federated simulation
# ---------------------------------------------------------------------------
NUM_CLIENTS = 4          # problem statement allows 3-5
SEEDS = [0, 1, 2]        # problem statement requires 3 random seeds
NUM_ROUNDS = 30          # FedAvg communication rounds
LOCAL_EPOCHS = 1         # passes over local data per round
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
FEDPROX_MU = 0.01        # only used by the extra FedProx experiment
PULSE_MIN_ACTIVE = 2     # PulseFed: at least this many clients talk every round
PULSE_MAX_SLEEP = 2      # PulseFed: no client sleeps more than this many rounds in a row

# Centralised training gets the same total number of passes over the data
# as FedAvg (NUM_ROUNDS * LOCAL_EPOCHS) so the comparison is fair.
CENTRAL_EPOCHS = NUM_ROUNDS * LOCAL_EPOCHS

# ---------------------------------------------------------------------------
# Model: a compact multilayer perceptron (MLP)
# ---------------------------------------------------------------------------
HIDDEN_SIZES = (64, 32)

# Plot settings
PLOT_SENSOR = 0          # which sensor to show in the actual-vs-predicted plot
PLOT_STEPS = 96 * 3      # 3 days of 15-minute steps
