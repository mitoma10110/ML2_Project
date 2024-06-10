import numpy as np
import pandas as pd
from copy import deepcopy

def convert_date_times(df:pd.DataFrame, columns:list) -> pd.DataFrame:
  '''
  Converts columns in a DataFrame to datetime format
  
  Inputs:
  - df: DataFrame containing the data
  - columns: List of columns to be converted
  
  Outputs:
  - DataFrame with the columns converted to datetime format
  '''
  for col in columns:
    df[str(col)] = pd.to_datetime(df[str(col)])
  return df

def extract_education(name:str) -> int:
  '''
  Maps strings found in names to their corresponding years of education
  Assuming none = 12 years of base education

  Inputs:
  - name: String containing the name of the person

  Outputs:
  - Integer representing the years of education
  '''
  if 'Bsc.' in name:
      return 15
  elif 'Msc.' in name:
      return 17
  elif 'Phd.' in name:
      return 19
  else:
      return 12


def custinfo_feature_eng(df:pd.DataFrame) -> pd.DataFrame:
  '''
  Creates new variables and adds them to df
  To be applied to the original customer_info dataset

  Inputs:
  - df: DataFrame containing the data

  Outputs:
  - DataFrame with the new variables added, and the old ones removed
  '''
  cust_info = deepcopy(df)
  cust_info.index = cust_info.customer_id
  cust_info.drop('customer_id', axis=1, inplace=True)

  # Fixing the negative percentages in the percentage_bought_promotion column
  cust_info['percentage_of_products_bought_promotion'] = abs(cust_info['percentage_of_products_bought_promotion'])

  # Extract age from birthdate (considering today=08/06/2024)
  convert_date_times(cust_info, columns=['customer_birthdate'])
  cust_info['age'] = cust_info['customer_birthdate'].apply(lambda x: (pd.to_datetime('2024-06-08 11:06:17.747865') - x).days // 365)
  cust_info.drop('customer_birthdate', axis=1, inplace=True)

  # Turn gender into a boolean (True if female)
  cust_info['gender'] = cust_info['customer_gender'].apply(lambda gender: 1 if gender == 'female' else 0)
  cust_info.drop('customer_gender', axis=1, inplace=True)

  # Turn card number into a boolean (True if the person has a card)
  cust_info['loyalty_program'] = cust_info['loyalty_card_number'].notnull()
  cust_info.drop('loyalty_card_number', axis=1, inplace=True)

  # Apply the function created above to create a new 'Education' column
  cust_info['education'] = cust_info['customer_name'].apply(extract_education)

  # Create a column to check if the person is vegetarian
  cust_info['vegetarian'] = np.where((cust_info['lifetime_spend_fish'] == 0) & (cust_info['lifetime_spend_meat'] == 0), 1, 0)

  # Create a lifetime_total_spent column
  cust_info['lifetime_total_spent'] = cust_info[['lifetime_spend_groceries', 'lifetime_spend_electronics', 'lifetime_spend_vegetables',
                                  'lifetime_spend_nonalcohol_drinks', 'lifetime_spend_alcohol_drinks', 'lifetime_spend_meat',
                                  'lifetime_spend_fish', 'lifetime_spend_hygiene', 'lifetime_spend_videogames',
                                  'lifetime_spend_petfood', 'lifetime_total_distinct_products']].sum(axis=1)

  print('Added columns: age, gender, loyalty_program, education, vegetarian, lifetime_total_spent.')
  print('Droped columns: customer_birthdate, customer_gender, loyalty_card_number.')
  print()
  return cust_info 