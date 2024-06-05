import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def plot_column_distribution(df: pd.DataFrame, column_name: str, filter_value=None):
    '''
    Plots the distribution of a specified column in a dataframe, optionally filtered
    
    Inputs: df - pandas dataframe
            column_name - (str) name of the column to plot
            filter_value - (int or float) optional value to filter the column data
            
    Outputs: Plot
    '''
    if filter_value is not None:
        filtered_data = df[df[column_name] > filter_value][column_name]
        plt.hist(filtered_data, bins=50)
        plt.xlabel(column_name)
        plt.ylabel('Frequency')
        plt.title(f'Distribution of {column_name} (Above {filter_value})')
        plt.show()
    else:
        plt.hist(df[column_name], bins=50)
        plt.xlabel(column_name)
        plt.ylabel('Frequency')
        plt.title(f'Distribution of {column_name}')
        plt.show()



def plot_variable_correlation(df: pd.DataFrame, cols_to_drop: list) -> list:
    '''
    Plots the correlation heatmap between variables, dropping the ones indicated
    '''
    # Generates the correlation matrix
    df_plot = df.drop(cols_to_drop, axis=1)
    df_corr = df_plot.corr()
    
    # Creates a heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(df_corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix')
    plt.show()

def plot_distributions_grid(df: pd.DataFrame, cols_to_drop: list, figsize=(20, 15), bins=30):
    """
    Plot a grid of distributions for all variables in the dataframe.

    Parameters:
    - dataframe: DataFrame containing data to plot.
    - figsize: Tuple specifying the figure size. Default is (20, 15).
    - bins: Number of bins for histograms. Default is 30.
    """
    # Drop the indicated columns
    dataframe = df.drop(cols_to_drop, axis=1)

    # Separate numerical and categorical columns
    numerical_cols = dataframe.select_dtypes(include=['number']).columns
    categorical_cols = dataframe.select_dtypes(include=['object']).columns

    # Calculate the number of rows and columns for the subplot grid
    total_cols = len(numerical_cols) + len(categorical_cols)
    grid_size = int(total_cols ** 0.5) + 1

    # Create subplots
    fig, axes = plt.subplots(nrows=grid_size, ncols=grid_size, figsize=figsize)
    axes = axes.flatten()

    # Plot numerical variables
    for i, numerical_col in enumerate(numerical_cols):
        ax = axes[i]
        sns.histplot(data=dataframe, x=numerical_col, bins=bins, ax=ax, kde=True)
        ax.set_title(f'Distribution of {numerical_col}')
        ax.set_xlabel('')
        ax.set_ylabel('Frequency')

    # Plot categorical variables
    for i, categorical_col in enumerate(categorical_cols, start=len(numerical_cols)):
        ax = axes[i]
        sns.countplot(data=dataframe, x=categorical_col, ax=ax)
        ax.set_title(f'Distribution of {categorical_col}')
        ax.set_xlabel('')
        ax.set_ylabel('Count')

    # Remove any unused subplots
    for j in range(total_cols, len(axes)):
        fig.delaxes(axes[j])

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()