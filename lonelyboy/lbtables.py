import pandas as import pd

def find_outlier_indices(df):
    quantiles = df.quantile([0.25, 0.75])
    feature_outlier_indices = []
    for feature in quantiles:
        q25, q75 = quantiles[feature].values
        iqr = q75 - q25
        
        outlier_indices = df.index[(df[feature] < q25 - (1.5 * iqr)) | (df[feature] > q75 + (1.5 * iqr))].tolist()
        feature_outlier_indices.append((feature, outlier_indices))
    return feature_outlier_indices

def drop_outliers(df, outliers):
    indices = set([])
    for col, col_indices in outliers:
        indices = indices.union(set(col_indices))
    return df.drop(list(indices))
