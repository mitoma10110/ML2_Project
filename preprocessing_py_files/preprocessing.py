import pandas as pd
import numpy as np
from copy import deepcopy
from sklearn.preprocessing import RobustScaler
from sklearn.impute import KNNImputer
from mlxtend.preprocessing import TransactionEncoder

# Preproc utils

def scaling(df:pd.DataFrame, numeric_cols:list) -> pd.DataFrame:
  '''
  Scales the given numeric columns using Robust
  '''
  scaler = RobustScaler()
  df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
  return df

def imputation(df:pd.DataFrame, colstoimpute:list) -> pd.DataFrame:
  '''
  Applies knnimputer to the given columns
  '''
  imputer = KNNImputer(n_neighbors=5)
  df[colstoimpute] = imputer.fit_transform(df[colstoimpute])
  return df

# Preproc functs
def scaling_imputation(df:pd.DataFrame) -> pd.DataFrame:
  '''
  Function that imputes missing values, scaling the data to do so
  '''
  
  # Drop non-numeric columns for imputation
  numeric_cols = df.select_dtypes(include=[np.number]).columns
  non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns

  # Scaling and imputation
  scaling(df, numeric_cols)
  imputation(df, numeric_cols)

  # Combine numeric and non-numeric columns back
  df = pd.concat([df[non_numeric_cols], df[numeric_cols]], axis=1)
  
  return df

def cust_basket_preproc(df):
    '''
    Function that applies preprocessing to the customer basket dataset
    '''
    cust_basket = deepcopy(df)
    # Convert purchases from strings to actual lists
    cust_basket['list_of_goods'] = cust_basket['list_of_goods'].apply(eval)

    cust_basket.index = cust_basket.invoice_id

    cust_basket.drop('invoice_id', axis=1, inplace=True)

    return cust_basket

def cust_basket_encoding(df):
  goods = df['list_of_goods'].to_list()

  te = TransactionEncoder()
  te_fit = te.fit(goods).transform(goods)
  encoded = pd.DataFrame(te_fit, columns=te.columns_)

  encoded_df = pd.concat([df.reset_index(drop=True), encoded], axis=1)
  encoded_df.index = df.index
  encoded_df.drop('list_of_goods', axis=1, inplace=True)

  return encoded_df