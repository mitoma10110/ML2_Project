import pandas as pd
import numpy as np

def float64_columns(df:pd.DataFrame) -> list:
    '''
    Function to return a list of column names with datatype 'float64' in a dataframe
    Inputs: df - dataframe
    Output: List of strings, columns with datatype 'float64'
    '''
    float64_cols = df.select_dtypes(include=['float64']).columns.tolist()
    return float64_cols


def float_converter(df:pd.DataFrame, variable_list:list) -> pd.DataFrame:
    '''
    Function to convert float columns in a dataframe to more memory efficient datatypes
    Inputs: df - dataframe
            variable_list - list of strings, names of specific columns in the dataframe
    Output: Dataframe with columns converted
    '''
    for var in variable_list:
        # Finds the minimum and maximum values in the column
        min_val = df[var].min(skipna=True)
        max_val = df[var].max(skipna=True)

        # Determines the appropriate datatype based on the range of values (finfo(dtype).max)
        float16_max = np.finfo('float16').max
        float32_max = np.finfo('float32').max
        
        if min_val > -float16_max and max_val < float16_max:
            df[var] = df[var].astype('float16')
        elif min_val > -float32_max and max_val < float32_max:
            df[var] = df[var].astype('float32')

    return df


def gender_coder(df:pd.DataFrame, gender_col:str) -> pd.DataFrame:
    '''
    Function to code the gender variable to boolean
    Input: df - dataframe
           gender_col - String, name of the gender column
    Output: Dataframe with changed column

    For gender, in this  dataset there are only 2 unique types
    The variable will be changed to 1 if female, 0 if male
    '''
    df['female'] = np.where(df[gender_col] == "female", 1, 0)
    df = df.drop(gender_col, axis=1).astype({'female' : "bool"})
    return df


def datetime_converter(df: pd.DataFrame, datetime_cols: list) -> pd.DataFrame:
    '''
    Function to convert the values in the columns listed in datetime_cols to
    the datatype 'datetime', ignoring missing values
    Input: df - Dataframe
           datetime_cols - List of strings, names of the columns to change
    Output: Dataframe with the changed columns
    '''
    for col in datetime_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    return df