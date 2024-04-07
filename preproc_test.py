# Import libraries ------------------
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import datasets -------------------
cust_info = pd.read_csv('data/customer_info.csv', index_col=2)
cust_basket = pd.read_csv('data/customer_basket.csv')

# Customer Info --------------------------------
cust_info.describe()