import pandas as pd

def selecting_by_correlation(df:pd.DataFrame, threshold=0.9) -> list:
    """
    Select features based on correlation to reduce multicollinearity.

    Inputs: df: pandas DataFrame, the dataset with features.
            threshold: float, the correlation threshold above which to consider features as highly correlated.

    Output: selected_features: list, the names of the selected features.
    """
    corr_matrix = df.corr().abs()

    # Pairs of highly correlated features
    high_corr_var = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if corr_matrix.iloc[i, j] > threshold:
                high_corr_var.append((corr_matrix.columns[i], corr_matrix.columns[j]))

    features_to_remove = set()

    for var1, var2 in high_corr_var:
        if var1 not in features_to_remove and var2 not in features_to_remove:
            # Removing the feature with higher overall correlation
            if corr_matrix[var1].sum() > corr_matrix[var2].sum():
                features_to_remove.add(var1)
            else:
                features_to_remove.add(var2)

    # Selected features = not removed
    selected_features = [feature for feature in df.columns if feature not in features_to_remove]

    return df[selected_features]
