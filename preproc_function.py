import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.impute import KNNImputer

def nulls_to_bool(df:pd.DataFrame, col_names:list) -> pd.DataFrame:
    '''
    Function that creates a new column for each column mentioned
    With the null values replaced with 0 and the non-null with 1
    Inputs: df (dataframe)
            col_names (list of strings) - columns to treat
    Output: df (dataframe) - altered dataframe
    '''

    for col in col_names:
        df[col + '_bool'] = df[col].notnull()

    return df

def knn_imputing(df:pd.DataFrame) -> pd.DataFrame:
    df_numeric = df.select_dtypes(include='number')
    indices = df_numeric.index
    
    # Fit and transform the imputer on the dataset
    # We use distance because the dataset has outliers
    imputer = KNNImputer(weights='distance', n_neighbors=20).fit(df_numeric)
    df_numeric_imputed = pd.DataFrame(imputer.transform(df_numeric), columns=df_numeric.columns, index=indices)

    # Add together the non-numerical original columns and the newly imputed ones
    df_imputed = pd.concat([df.drop(columns=df_numeric.columns), df_numeric_imputed], axis=1)

    return df_imputed

def float64_columns(df:pd.DataFrame) -> list:
    '''
    Function to return a list of column names with datatype 'float64' in a dataframe
    Inputs: df (dataframe)
    Output: float64_cols (list of strings) - column names with datatype 'float64'
    '''
    float64_cols = df.select_dtypes(include=['float64']).columns.tolist()
    return float64_cols

def float_converter(df:pd.DataFrame, variable_list:list) -> pd.DataFrame:
    '''
    Function to convert float columns in a DataFrame to more memory-efficient float types
    Inputs: df (dataframe)
            variable_list (list of strings) - names of specific columns in the DataFrame
    Output: DataFrame with columns converted to more memory-efficient float data types
    '''
    for var in variable_list:
        # Finds the minimum and maximum values in the column
        min_val = df[var].min(skipna=True)
        max_val = df[var].max(skipna=True)

        # Determine the appropriate float type based on the range of values
        float16_max = np.finfo('float16').max
        float32_max = np.finfo('float32').max
        
        if min_val > -float16_max and max_val < float16_max:
            df[var] = df[var].astype('float16')
        elif min_val > -float32_max and max_val < float32_max:
            df[var] = df[var].astype('float32')

    return df

def gender_coder(df:pd.DataFrame, gender_col:str) -> pd.DataFrame:
    '''
    Function that takes a gender column and changes "male" to 0 and "female" to 1
    Input:  df (dataframe)
            gender_col (string) - name of the gender column
    Output: df (dataframe) - altered dataframe
    '''
    df[gender_col] = np.where(df[gender_col] == "male", 0, 1)
    df = df.astype({gender_col : "bool"})
    return df

def datetime_converter(df: pd.DataFrame, datetime_cols: list) -> pd.DataFrame:
    '''
    Function that changes the dtype of the listed columns to datetime
    Input:  df (dataframe)
            datetime_cols (list of strings) - name of the datetime columns
    Output: df (dataframe) - altered dataframe
    '''
    for col in datetime_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    return df

def preproc_applier(df:pd.DataFrame, datetime_cols:list=None, gender_col:str=None) -> pd.DataFrame:
    '''
    Function that applies preprocessing functions to a dataframe
    Input:  df (dataframe)
            datetime_cols (list of strings) - column names to change to datetime
            gender_col (string) - name of the gender column
    Output: df (dataframe) - altered dataframe
    '''
    df = datetime_converter(df, datetime_cols)
    float64_cols = float64_columns(df)
    df = float_converter(df, float64_cols)
    df = gender_coder(df, gender_col)
    
    return df