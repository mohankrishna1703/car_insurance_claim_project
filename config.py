# config.py
import os

# Reproducibility
RANDOM_SEED = 42

# Data paths
DATA_DIR = os.path.join(os.getcwd(), "data")
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")

# Output
SUBMISSION_PATH = os.path.join(os.getcwd(), "submission.csv")
MODEL_PATH = os.path.join(os.getcwd(), "final_model.joblib")