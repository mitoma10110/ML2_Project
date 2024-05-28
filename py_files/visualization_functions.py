import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def plot_column_distribution(df:pd.DataFrame, column_name:str):
    '''
    Function that plots the distribution of the given column
    Inputs: df (dataframe)
            column_name (string) - column to plot
    Outputs: Plot
    '''
    
    # Counts the instances of each value (needed for categorical columns)
    column_counts = df[column_name].value_counts()

    # Plot
    plt.bar(column_counts.index, column_counts.values)
    plt.xlabel(column_name)
    plt.ylabel('Counts')
    plt.title(f'Counts of {column_name}')
    plt.show()

def variable_correlation(df: pd.DataFrame, threshold: float = 0.7) -> list:
    # Generates the correlation matrix
    correlation_matrix = df.drop(['customer_name','typical_hour','year_first_transaction','customer_gender', 'customer_birthdate'], axis=1).corr()
    
    # Creates a heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix')
    plt.show()

    # Finds highly correlated variables (below -0.7 or above 0.7, not including)
    correlated_vars = list()
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > threshold:
                colname_i = correlation_matrix.columns[i]
                colname_j = correlation_matrix.columns[j]
                correlated_vars.append(colname_i)
                correlated_vars.append(colname_j)
                correlated_vars.append(' / ')

    print('Highly correlated variables: ', correlated_vars)
