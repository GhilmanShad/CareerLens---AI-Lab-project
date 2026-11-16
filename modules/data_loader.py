import pandas as pd
import os

def load_and_inspect_data(filepath="data/job_postings.csv"):
    if not os.path.exists(filepath):
        return "Error: Dataset file nahi mili. Please check the path."
    
    # Data load karo
    df = pd.read_csv(filepath)
    
    print("--- Dataset Shape ---")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    
    print("\n--- Columns & Data Types ---")
    print(df.dtypes)
    
    print("\n--- Missing Values ---")
    print(df.isnull().sum())
    
    print("\n--- Duplicate Rows ---")
    print(f"Total duplicates: {df.duplicated().sum()}")
    
    return df

if __name__ == "__main__":
    df = load_and_inspect_data()