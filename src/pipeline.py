"""Builds the full preprocessing + estimator sklearn Pipeline.

All fitting (StandardScaler, OneHotEncoder, CountVectorizer) happens inside
the sklearn Pipeline/ColumnTransformer, so calling `.fit` only on a training
fold and `.transform`/`.predict` on the corresponding validation/test fold
(as scikit-learn's cross_validate / GridSearchCV do automatically) guarantees
no leakage between folds.
"""

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features import FeatureEngineer
from src.config import TARGET_COL, DROP_COLS

NUMERIC_COLS = [
    "BaseOfCode", "BaseOfData", "Characteristics", "DllCharacteristics",
    "Entropy", "FileAlignment", "ImageBase", "NumberOfRvaAndSizes",
    "NumberOfSections", "NumberOfSymbols", "PointerToSymbolTable", "Size",
    "SizeOfCode", "SizeOfHeaders", "SizeOfImage", "SizeOfInitializedData",
    "SizeOfOptionalHeader", "SizeOfUninitializedData", "TimeDateStamp",
]

ENGINEERED_NUMERIC_COLS = [
    "identify_missing", "has_upx", "has_delphi", "has_dotnet", "has_msvc",
    "has_installer", "has_packer_generic", "num_imported_dlls",
    "num_imported_symbols",
]

CATEGORICAL_COLS = ["Machine"]

TEXT_COLS = {"ImportedDlls": 40, "ImportedSymbols": 60}


def _whitespace_tokenizer(text: str):
    return text.split()


def build_column_transformer() -> ColumnTransformer:
    transformers = [
        ("numeric", StandardScaler(), NUMERIC_COLS + ENGINEERED_NUMERIC_COLS),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
    ]
    for col, max_features in TEXT_COLS.items():
        transformers.append((
            f"text_{col}",
            CountVectorizer(
                tokenizer=_whitespace_tokenizer,
                token_pattern=None,
                lowercase=False,
                binary=True,
                max_features=max_features,
            ),
            col,
        ))
    return ColumnTransformer(transformers, remainder="drop")


def build_preprocessing_pipeline() -> Pipeline:
    """Returns the feature-engineering + column-transform pipeline (no
    estimator attached). Append a final classifier step to use for training.
    """
    return Pipeline([
        ("feature_engineer", FeatureEngineer()),
        ("column_transform", build_column_transformer()),
    ])


def build_full_pipeline(estimator) -> Pipeline:
    """Returns the complete pipeline: feature engineering + column
    transforms + the given estimator as the final step.
    """
    return Pipeline([
        ("feature_engineer", FeatureEngineer()),
        ("column_transform", build_column_transformer()),
        ("classifier", estimator),
    ])


def split_X_y(df):
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return X, y
