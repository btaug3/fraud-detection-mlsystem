"""Central configuration: paths, feature lists, split fractions, cost model."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # project root folder
DATA_RAW = ROOT / "data" / "raw" / "creditcard.csv"
DATA_SAMPLE = ROOT / "data" / "sample"
ARTIFACTS = ROOT / "artifacts"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

MODEL_PATH = ARTIFACTS / "fraud_model.joblib"
BASELINE_PATH = ARTIFACTS / "baseline_logreg.joblib"
METADATA_PATH = ARTIFACTS / "model_metadata.json"
REFERENCE_PROFILE_PATH = ARTIFACTS / "reference_profile.json"

ID_COL = "transaction_id"
TARGET = "Class"
TIME_COL = "Time"
PCA_COLS = [f"V{i}" for i in range(1, 29)]
RAW_FEATURES = [TIME_COL, *PCA_COLS, "Amount"]                      # what the API accepts (31)
MODEL_FEATURES = [*PCA_COLS, "log_amount", "hour_sin", "hour_cos"]  # what the model sees (31)

TRAIN_FRAC, VAL_FRAC = 0.70, 0.15     # remaining 0.15 = test

# Business cost model (EUR) - used to choose the decision threshold
COST_FP = 5.0          # false alarm: analyst review + customer friction
COST_FN_FIXED = 25.0   # missed fraud: transaction amount lost + this handling fee

RANDOM_STATE = 42