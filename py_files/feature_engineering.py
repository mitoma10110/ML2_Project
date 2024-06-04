import numpy as np
import pandas as pd

def extract_education(name):
    if 'Bsc.' in name:
        return 'Bsc.'
    elif 'Msc.' in name:
        return 'Msc.'
    elif 'Phd.' in name:
        return 'Phd.'
    else:
        return 'None'

def custinfo_create_variables(df):
    '''
    Creates new variables and adds them to df
    '''
    # Extract age from birthdate
    df['age'] = df['customer_birthdate'].apply(lambda x: (pd.Timestamp.now() - x).days // 365)
    df.drop('customer_birthdate', axis=1, inplace=True)

    # Turn gender into a boolean (True if female)
    df['gender'] = df['customer_gender'].apply(lambda gender: 1 if gender == 'female' else 0)
    df.drop('customer_gender', axis=1, inplace=True)

    # Turn card number into a boolean (True if the person has a card)
    df['loyalty_program'] = df['loyalty_card_number'].notnull()
    df.drop('loyalty_card_number', axis=1, inplace=True)

    # Apply the function to create a new 'Education' column
    df['education'] = df['customer_name'].apply(extract_education)

    # Create a lifetime_total_spent column
    df['lifetime_total_spent'] = df[['lifetime_spend_groceries', 'lifetime_spend_electronics', 'lifetime_spend_vegetables',
                                    'lifetime_spend_nonalcohol_drinks', 'lifetime_spend_alcohol_drinks', 'lifetime_spend_meat', 
                                    'lifetime_spend_fish', 'lifetime_spend_hygiene', 'lifetime_spend_videogames', 
                                    'lifetime_spend_petfood', 'lifetime_total_distinct_products']].sum(axis=1)

    return df