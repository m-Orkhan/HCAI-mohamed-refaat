import numpy as np
from scipy.optimize import minimize

def plackett_luce_neg_log_likelihood(w, rankings, l2=1.0):
    total = 0.0
    for X, k in rankings:
        utilities = X @ w
        for i in range(k):
            remaining = utilities[i:]
            m = remaining.max()
            total -= remaining[0] - (m + np.log(np.exp(remaining - m).sum()))
    total += 0.5 * l2 * np.dot(w, w)
    return total

def fit_preferences(rankings, n_features, l2=1.0):
    if not rankings:
        return np.zeros(n_features)
    result = minimize(
        plackett_luce_neg_log_likelihood, np.zeros(n_features),
        args=(rankings, l2), method='L-BFGS-B'
    )
    return result.x

def predict_pairwise(w, x_a, x_b):
    return 1.0 / (1.0 + np.exp(-((x_a - x_b) @ w)))