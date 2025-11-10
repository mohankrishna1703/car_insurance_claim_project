import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from load_data import load_data

def perform_eda():
    # Step 1: Load data
    df = load_data("train_data")
    print(f"\nLoaded {len(df)} rows for EDA.\n")

    # Create 'plots' folder if not exists
    os.makedirs("plots", exist_ok=True)

    # Step 2: Basic info
    print("Dataset Info:")
    print(df.info())

    print("\nStatistical Summary:")
    print(df.describe())

    # Check for missing values
    missing = df.isnull().sum()
    print("\nMissing Values per Column:")
    print(missing[missing > 0])

    # Claim Distribution Plot
    if "is_claim" in df.columns:
        plt.figure(figsize=(5, 4))
        sns.countplot(x="is_claim", data=df, palette="Set2")
        plt.title("Claim Distribution")
        plt.xlabel("Is Claim (0 = No, 1 = Yes)")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig("plots/eda_claim_distribution.png")
        plt.close()
        print("Saved: plots/eda_claim_distribution.png")

    # Correlation Heatmap (numeric only)
    numeric_cols = df.select_dtypes(include=["int64", "float64"])
    if not numeric_cols.empty:
        plt.figure(figsize=(10, 8))
        sns.heatmap(numeric_cols.corr(), cmap="coolwarm", annot=False)
        plt.title("Feature Correlation Heatmap")
        plt.tight_layout()
        plt.savefig("plots/eda_correlation_heatmap.png")
        plt.close()
        print("Saved: plots/eda_correlation_heatmap.png")

    # Distribution of numeric features
    num_cols = numeric_cols.columns[:5]  # only a few to keep simple
    df[num_cols].hist(figsize=(10, 8), bins=20, color="skyblue")
    plt.suptitle("Numeric Feature Distributions")
    plt.tight_layout()
    plt.savefig("plots/eda_numeric_distributions.png")
    plt.close()
    print("Saved: plots/eda_numeric_distributions.png")

if __name__ == "__main__":
    perform_eda()