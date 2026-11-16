import numpy as np
import pandas as pd

# Tree ka Node structure
class TreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value # Leaf node ki prediction

# Gini Impurity calculate karo
def calculate_gini(y):
    if len(y) == 0:
        return 0
    classes, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    gini = 1 - np.sum(probabilities ** 2)
    return gini

# Best split dhoondho
def get_best_split(X, y):
    best_gini = float('inf')
    best_feature = None
    best_threshold = None

    for feature_idx in range(X.shape[1]):
        thresholds = np.unique(X[:, feature_idx])
        for t in thresholds:
            left_mask = X[:, feature_idx] <= t
            right_mask = X[:, feature_idx] > t
            
            if len(y[left_mask]) == 0 or len(y[right_mask]) == 0:
                continue
                
            gini_left = calculate_gini(y[left_mask])
            gini_right = calculate_gini(y[right_mask])
            
            # Weighted Gini
            weighted_gini = (len(y[left_mask]) / len(y)) * gini_left + (len(y[right_mask]) / len(y)) * gini_right
            
            if weighted_gini < best_gini:
                best_gini = weighted_gini
                best_feature = feature_idx
                best_threshold = t
                
    return best_feature, best_threshold

# Recursive Tree Build
def build_tree(X, y, depth=0, max_depth=3):
    # Stopping criteria: Agar sab same class hain ya depth limit cross ho gayi
    if len(np.unique(y)) == 1 or depth >= max_depth:
        return TreeNode(value=np.unique(y)[0])

    best_feature, best_threshold = get_best_split(X, y)
    
    if best_feature is None:
        return TreeNode(value=np.unique(y)[0])

    left_mask = X[:, best_feature] <= best_threshold
    right_mask = X[:, best_feature] > best_threshold
    
    left_node = build_tree(X[left_mask], y[left_mask], depth + 1, max_depth)
    right_node = build_tree(X[right_mask], y[right_mask], depth + 1, max_depth)
    
    return TreeNode(feature=best_feature, threshold=best_threshold, left=left_node, right=right_node)

# Prediction function
def predict(node, x):
    if node.value is not None:
        return node.value
    if x[node.feature] <= node.threshold:
        return predict(node.left, x)
    else:
        return predict(node.right, x)