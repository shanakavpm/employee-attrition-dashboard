"""Central project paths and modelling settings."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "Saudi Employee Attrition Dataset"
    / "Original Dataset of Employee Attrition.xlsx"
)
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "employee_attrition_cleaned.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

TARGET_COLUMN = "attrition"
ID_COLUMN = "id"
RANDOM_STATE = 42
TEST_SIZE = 0.25
CV_FOLDS = 5
MIN_GROUP_SIZE = 15
