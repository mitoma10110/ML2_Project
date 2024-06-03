import pandas as pd

def custinfo_feature_selection(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Function that removes unwanted variables from the dataset
    '''
    highly_correlated_vars = ['lifetime_spend_videogames', 'lifetime_spend_meat']
    for var in highly_correlated_vars:
        if var in df.columns:
            df.drop(var, axis=1, inplace=True)

    return df