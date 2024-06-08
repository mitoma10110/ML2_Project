import pandas as pd
import numpy as np
from copy import deepcopy
from sklearn.preprocessing import RobustScaler
from sklearn.impute import KNNImputer
from preprocessing_py_files.feature_engineering import *

def scaling(df:pd.DataFrame, numeric_cols) -> pd.DataFrame:
    '''
    Assuming df is numeric
    '''
    scaler = RobustScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

def convert_date_times(df, columns):
    for col in columns:
        df[str(col)] = pd.to_datetime(df[str(col)])


def imputation(df, numeric_cols):
    imputer = KNNImputer(n_neighbors=5)
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])


def cust_info_preproc(df:pd.DataFrame) -> pd.DataFrame:
    '''
    Function that applies preprocessing to the customer info dataset
    '''
    cust_info = deepcopy(df)

    # Convert birth date to datetime
    convert_date_times(cust_info, columns=['customer_birthdate'])

    # Feature engineering
    cust_info = custinfo_create_variables(cust_info)

    # Drop non-numeric columns for imputation
    numeric_cols = cust_info.select_dtypes(include=[np.number]).columns
    non_numeric_cols = cust_info.select_dtypes(exclude=[np.number]).columns


    scaling(cust_info, numeric_cols)
    imputation(cust_info, numeric_cols)


    # Missing values: KNN Imputation
    imputer = KNNImputer(n_neighbors=5)
    cust_info[numeric_cols] = imputer.fit_transform(cust_info[numeric_cols])
    
    # Combine numeric and non-numeric columns back
    cust_info = pd.concat([cust_info[non_numeric_cols], cust_info[numeric_cols]], axis=1)

    return cust_info

def cust_basket_preproc(df, product_maping):
    '''
    Function that applies preprocessing to the customer basket dataset
    '''
    cust_basket = deepcopy(df)
    # Convert purchases from strings to actual lists
    cust_basket['list_of_goods'] = cust_basket['list_of_goods'].apply(eval)

    # Create a dictionary from product mapping
    product_mapping_dict = pd.Series(product_mapping['category'].values, index=product_mapping['product_name']).to_dict()

    # Map each item in the list_of_goods to its category
    cust_basket['categories'] = cust_basket['list_of_goods'].apply(lambda x: [product_mapping_dict.get(item, 'Unknown') for item in x])

    # Create the category columns with counts of items purchased in each category
    for category in product_mapping['category'].unique():
        cust_basket[category] = cust_basket['categories'].apply(lambda x: x.count(category))

    # Drop the intermediate 'categories' column
    cust_basket = cust_basket.drop(columns=['categories'])

    return cust_basket


def modelling_separator(df:pd.DataFrame):
    '''
    Function that separates the variables into variables to use for
    modelling and for interpreting the clusters
    '''
    training_vars = ['customer_birthdate', 'kids_home', 'teens_home', 'number_complaints',
                     'distinct_stores_visited', 'lifetime_spend_groceries', 'lifetime_spend_electronics',
                     'typical_hour', 'lifetime_spend_vegetables', 'lifetime_spend_nonalcohol_drinks',
                     'lifetime_spend_alcohol_drinks', 'lifetime_spend_meat', 'lifetime_spend_fish',
                     'lifetime_spend_hygiene', 'lifetime_spend_videogames', 'lifetime_spend_petfood',
                     'lifetime_total_distinct_products', 'percentage_of_products_bought_promotion', 'year_first_transaction']
    training_set = pd.DataFrame

    for var in training_vars:
        if var in df.columns:
            training_set[var] = df[var]

    return training_set

