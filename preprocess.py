import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from category_encoders import TargetEncoder

def build_preprocessor(X):
    """Build preprocessing pipeline for numeric and categorical features."""

    # Identify feature types
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

    # Remove target column if present
    if "is_claim" in numeric_cols:
        numeric_cols.remove("is_claim")

    # Split categorical into low and high cardinality
    low_card = [col for col in categorical_cols if X[col].nunique() <= 10]
    high_card = [col for col in categorical_cols if X[col].nunique() > 10]

    print("Low-card categorical:", low_card[:5], "...")
    print("High-card categorical:", high_card[:5], "...")

    # Transformers
    numeric_transformer = StandardScaler()
    low_card_transformer = OneHotEncoder(handle_unknown="ignore")
    high_card_transformer = TargetEncoder()

    # Column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("low_cat", low_card_transformer, low_card),
            ("high_cat", high_card_transformer, high_card)
        ]
    )

    return preprocessor