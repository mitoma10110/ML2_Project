import pandas as pd

def feature_selection(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Function that removes unwanted variables from the dataset
    '''
    highly_correlated_vars = ['lifetime_spend_videogames', 'lifetime_spend_meat']
    for var in highly_correlated_vars:
        if var in df.columns:
            df.drop(var, axis=1, inplace=True)

    return df

def modelling_separator(df:pd.DataFrame):
  '''
  Function that separates out the columns to model
  '''

  vars = ['lifetime_spend_groceries', 'lifetime_spend_electronics', 'lifetime_spend_vegetables',
          'lifetime_spend_nonalcohol_drinks', 'lifetime_spend_alcohol_drinks', 'lifetime_spend_meat',
          'lifetime_spend_fish', 'lifetime_spend_hygiene', 'lifetime_spend_videogames',
          'lifetime_spend_petfood', 'lifetime_total_distinct_products',
          'percentage_products_bought_promotion', 'lifetime_total_spent',
          'education', 'kids_home', 'teens_home', 'age',
          'year_first_transaction', 'number_complaints', 'typical_hour', 'distinct_stores_visited']

  modelling_sol = pd.DataFrame()

  # Add columns to purchaseHistorySolution if they exist in the input DataFrame
  for var in vars:
      if var in df.columns:
          modelling_sol[var] = df[var]

  return modelling_sol