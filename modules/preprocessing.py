import pandas as pd
import numpy as np

def clean_data(df):
    # 1. Duplicate rows hatao
    df = df.drop_duplicates()
    
    # 2. Missing values handle karo (Important columns ke liye)
    # Agar job_title ya company missing hai, toh us row ka faida nahi
    df = df.dropna(subset=['job_title', 'company'])
    
    # 3. Salary ko numeric banao (agar string mein hai toh)
    if 'salary' in df.columns:
        df['salary'] = pd.to_numeric(df['salary'], errors='coerce')
        
    # 4. Missing salary ko median se fill karo (ya drop kar sakte hain, par abhi fill kar rahe hain)
    if 'salary' in df.columns:
        median_salary = df['salary'].median()
        df['salary'] = df['salary'].fillna(median_salary)
        
    return df