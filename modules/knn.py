import numpy as np
import pandas as pd

def standardize_series(series):
    mean = np.mean(series)
    std = np.std(series)
    if std == 0:
        return series - mean # Agar sab same hain
    return (series - mean) / std

def get_knn_recommendations(df, query_salary, k=3):
    # EDGE CASE FIX: Agar K dataset se bada ho jaye
    if k > len(df):
        k = len(df)
    # 1. Data ko scale karo (Standardization)
    df_scaled = df.copy()
    mean_salary = np.mean(df['salary'])
    std_salary = np.std(df['salary'])
    
    if std_salary == 0:
         df_scaled['salary_scaled'] = 0
         query_scaled = 0
    else:
        df_scaled['salary_scaled'] = (df['salary'] - mean_salary) / std_salary
        query_scaled = (query_salary - mean_salary) / std_salary
    
    # 2. Manual Euclidean Distance calculate karo
    # Formula: sqrt((x2 - x1)^2 + (y2 - y1)^2 ...) 
    # Hamare paas sirf 1 feature (salary) hai, toh distance = abs(x2 - x1)
    df_scaled['distance'] = np.abs(df_scaled['salary_scaled'] - query_scaled)
    
    # 3. Distance ke hisaab se sort karo aur Top K return karo
    nearest_jobs = df_scaled.sort_values(by='distance').head(k)
    
    return nearest_jobs