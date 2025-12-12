# imputer/evaluate.py
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_imputation(imputed, ground_truth, mask_observed, feature_names):
    missing = ~mask_observed
    y_true = ground_truth[missing]
    y_pred = imputed[missing]
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    per_feat = {}
    for j, name in enumerate(feature_names):
        idx = missing[:, j]
        if idx.sum() > 0:
            per_feat[name] = math.sqrt(mean_squared_error(ground_truth[idx, j], imputed[idx, j]))
        else:
            per_feat[name] = None
    return rmse, mae, r2, per_feat
