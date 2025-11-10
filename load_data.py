# Loads data from SQL database into Python
import sqlite3
import pandas as pd

def load_data(table_name):
    conn = sqlite3.connect("data/database.db")
    df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
    conn.close()
    print(f"Loaded {len(df)} rows from {table_name}")
    return df

if __name__ == "__main__":
    df = load_data("train_data")
    print(df.head())