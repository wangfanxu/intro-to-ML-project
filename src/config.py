"""Shared configuration constants for the malware detection project."""

import os

RANDOM_STATE = 42

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

RAW_CSV = os.path.join(DATA_DIR, "brazilian-malware.csv")
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")

TARGET_COL = "Label"

# Columns dropped prior to modeling:
# - SHA1: unique file identifier, not predictive, high-cardinality leakage risk.
# - FirstSeenDate: metadata about when sample was first seen, not a static
#   property of the executable itself (leaks collection-time information).
# - Magic, PE_TYPE: constant/zero-variance in this dataset (single unique
#   value each), so they carry no predictive signal.
DROP_COLS = ["SHA1", "FirstSeenDate", "Magic", "PE_TYPE"]

# Fixed, human-curated keyword list used to derive binary flags from the
# free-text 'Identify' (packer/compiler signature) column. These keywords are
# domain knowledge constants, not statistics fit from the data, so applying
# them does not leak information between CV folds / train-test split.
IDENTIFY_KEYWORDS = {
    "has_upx": "upx",
    "has_delphi": "delphi",
    "has_dotnet": ".net",
    "has_msvc": "visual c++",
    "has_installer": "installshield",
    "has_packer_generic": "protector",
}

# Free-text/list-like columns requiring custom feature engineering
# (space-separated tokens) rather than plain one-hot encoding.
LIST_COLS = ["ImportedDlls", "ImportedSymbols"]

# Free-text categorical column with many unique values and ~28% missing
# (packer/identification signature string).
TEXT_CATEGORICAL_COLS = ["Identify"]

TEST_SIZE = 0.2
N_SPLITS = 10

# Full ordered list of the 27 raw input columns present in the source CSV
# (everything except the target 'Label'). Used to build the manual-entry
# web form and to validate uploaded batch-prediction files.
ALL_INPUT_COLS = [
    "BaseOfCode", "BaseOfData", "Characteristics", "DllCharacteristics",
    "Entropy", "FileAlignment", "FirstSeenDate", "Identify", "ImageBase",
    "ImportedDlls", "ImportedSymbols", "Machine", "Magic",
    "NumberOfRvaAndSizes", "NumberOfSections", "NumberOfSymbols", "PE_TYPE",
    "PointerToSymbolTable", "SHA1", "Size", "SizeOfCode", "SizeOfHeaders",
    "SizeOfImage", "SizeOfInitializedData", "SizeOfOptionalHeader",
    "SizeOfUninitializedData", "TimeDateStamp",
]

DEMO_SAMPLE_CSV = os.path.join(DATA_DIR, "demo_test_sample.csv")
