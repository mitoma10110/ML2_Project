import matplotlib.pyplot as plt
import pandas as pd

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
