import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def run_eda():
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")

    print("Train shape:", train.shape)
    print("Test shape:", test.shape)

    print("\nTrain columns:\n", train.columns.tolist())

    # Target balance
    target_dist = train["is_claim"].value_counts(normalize=True) * 100
    print("\nTarget balance:\n", target_dist)

    # Save target distribution plot
    plt.figure(figsize=(6,4))
    sns.countplot(x="is_claim", data=train)
    plt.title("Target Distribution (is_claim)")
    plt.savefig("eda_target_distribution.png")
    plt.close()

    # Numeric vs target
    numeric_cols = train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    for col in numeric_cols[:10]:
        plt.figure(figsize=(6,4))
        sns.histplot(data=train, x=col, hue="is_claim", kde=True, bins=30)
        plt.title(f"{col} vs is_claim")
        plt.savefig(f"eda_{col}_vs_claim.png")
        plt.close()

    print("\nEDA complete. Figures saved as PNG files in project folder.")

if __name__ == "__main__":
    run_eda()