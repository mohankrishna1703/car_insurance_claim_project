from train_model import train

def main():
    print("🚀 Starting training pipeline...")
    best_model, best_thresh = train()
    print("\n✅ Training complete!")
    print(f"Best threshold chosen: {best_thresh}")
    print("Final model saved as final_model.joblib, and submission.csv is ready.")

if __name__ == "__main__":
    main()