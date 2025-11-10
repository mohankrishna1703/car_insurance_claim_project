import sqlite3
import pandas as pd
import os

def create_database():
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect("data/database.db")

    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")

    train_df.to_sql("train_data", conn, if_exists="replace", index=False)
    test_df.to_sql("test_data", conn, if_exists="replace", index=False)

    conn.close()
    print("Database created successfully.")

if __name__ == "__main__":
    create_database()