import numpy as np

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
    Creates new variables and adds them to cust_info
    '''
    df['vegetarian'] = np.where((df['lifetime_spend_meat'] == 0) & 
                                (df['lifetime_spend_fish'] == 0), 
                                1, 0)

    # Apply the function to create a new 'Education' column
    df['education'] = df['customer_name'].apply(extract_education)

    return df