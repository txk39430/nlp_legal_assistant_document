from pathlib import Path

# Root of the project directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Common directories
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

# Make sure the directories exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# ---- Dataset & model settings ----

# We have started with the California subset of the BillSum dataset
# (legal bills + human-written summaries).
BILLSUM_SPLIT = "ca_test"  # Splits it into train/val ourselves

# Summarization model
SUMMARIZATION_MODEL_NAME = "t5-small"

# Sequence length limits 
MAX_INPUT_LENGTH = 1024   # tokens for bill text
MAX_TARGET_LENGTH = 256   # tokens for summary


# ---- QA settings ----

# A strong baseline QA model fine-tuned on SQuAD2
QA_MODEL_NAME = "deepset/roberta-base-squad2"

# Max tokens for QA context (model limit is usually 512)
QA_MAX_CONTEXT_LENGTH = 512



