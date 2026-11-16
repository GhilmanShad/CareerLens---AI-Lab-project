import numpy as np
import pandas as pd

def run_kmeans(df, column='salary', k=3, max_iters=100):
    # Data ko NumPy array mein le lo
    data = df[column].values.reshape(-1, 1)

     # EDGE CASE FIX: Agar K unique salaries se zyada ho jaye
    unique_salaries = len(np.unique(data))
    if k > unique_salaries:
        k = unique_salaries
    
    # 1. Centroids initialize karo (Randomly data points mein se k utha lo)
    np.random.seed(42) # Taake har baar same result aaye
    random_indices = np.random.choice(len(data), size=k, replace=False)
    centroids = data[random_indices]
    
    labels = np.zeros(len(data))
    
    for i in range(max_iters):
        # 2. Har point ka distance nikaalo sabhi centroids se
        distances = np.abs(data - centroids.T) # 1D array ke liye abs difference
        
        # 3. Har point ko us centroid mein assign karo jo sabse qareeb ho
        new_labels = np.argmin(distances, axis=1)
        
        # 4. Check karo ke labels change hue ya nahi (convergence)
        if np.array_equal(labels, new_labels):
            print(f"K-Means converged at iteration {i}")
            break
            
        labels = new_labels
        
        # 5. Naye centroids calculate karo (us cluster ke sabhi points ka mean)
        for j in range(k):
            cluster_points = data[labels == j]
            if len(cluster_points) > 0:
                centroids[j] = np.mean(cluster_points)
                
    # Result ko dataframe mein add karo
    result_df = df.copy()
    result_df['cluster'] = labels.astype(int)
    
    return result_df, centroids