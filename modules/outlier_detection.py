import numpy as np
import pandas as pd

def detect_zscore_outliers(df, column='salary', threshold=3.0):
    if column not in df.columns:
        return pd.DataFrame(), 0
    
    # NumPy use karke mean aur std calculate karo
    data = df[column].values
    mean = np.mean(data)
    std = np.std(data)
    
    # Agar standard deviation 0 hai (sab salaries same hain), toh division error se bacho
    if std == 0:
        return pd.DataFrame(), 0
        
    # Z-score formula: (x - mean) / std
    z_scores = (data - mean) / std
    
    # Jahan Z-score > threshold ya < -threshold ho, unko flag karo
    outliers = df[np.abs(z_scores) > threshold].copy()
    outliers['z_score'] = z_scores[np.abs(z_scores) > threshold]
    
    return outliers, len(outliers)

def detect_iqr_outliers(df, column='salary'):
    if column not in df.columns:
        return pd.DataFrame(), {}
    
    data = df[column].values
    
    # Q1 (25th percentile), Q2 (Median), Q3 (75th percentile) calculate karo
    Q1 = np.percentile(data, 25)
    Q2 = np.percentile(data, 50)
    Q3 = np.percentile(data, 75)
    
    # IQR calculate karo
    IQR = Q3 - Q1
    
    # Fences (Boundaries) set karo
    lower_fence = Q1 - 1.5 * IQR
    upper_fence = Q3 + 1.5 * IQR
    
    # Jahan salary fence se bahar ho, unko flag karo
    outliers = df[(df[column] < lower_fence) | (df[column] > upper_fence)].copy()
    
    stats = {
        "Q1": Q1, "Median": Q2, "Q3": Q3, "IQR": IQR,
        "Lower Fence": lower_fence, "Upper Fence": upper_fence
    }
    
    return outliers, stats