import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compare(df_one, df_two):

    df_one_means = df_one.select_dtypes(include='number').mean().round()
    df_two_means = df_two.select_dtypes(include='number').mean().round()

    comparison_df = pd.concat([df_one_means, df_two_means], axis=1)
    comparison_df.columns = ['df1_means', 'df2_means']

    return comparison_df

def find_fishermen(data, longitude_threshold=-9.45, tolerance=0.1):
    """
    Function to find fishermen based on their longitude being near a specified threshold.

    Parameters:
    - data: DataFrame containing latitude and longitude columns.
    - longitude_threshold: Longitude threshold to determine if a point is near a fisherman's location.
    - tolerance: Tolerance value for determining proximity.

    Returns:
    - DataFrame containing only fishermen.
    """
    # Filter data based on longitude
    fishermen = data[(data['longitude'] >= longitude_threshold - tolerance) & 
                     (data['longitude'] <= longitude_threshold + tolerance)]
    data = data.drop(fisherman for fisherman in fishermen.index)
    
    return data, fishermen


def show_outliers(data:pd.DataFrame, threshold=2) -> dict:
    """
    Function to find outliers in each column of a DataFrame and print the number of outliers.

    Parameters:
    - data: DataFrame containing the data.
    - threshold: Threshold for determining outliers, default is 2 standard deviations.

    Returns:
    - A dictionary containing the number of outliers for each column.
    """
    outlier_counts = {}
    
    for column in data.columns:
       # if data[column].dtype.kind in 'bifc':  # Check if the column is of numeric type
            datacopy = data.copy()
            mean = datacopy[column].mean()
            std = datacopy[column].std()
            outliers = datacopy[(datacopy[column] > mean + threshold * std) | (datacopy[column] < mean - threshold * std)]
            outlier_counts[column] = len(outliers)
    
    for column, count in outlier_counts.items():
        print(f"Column '{column}' has {count} outliers.")
    
    return outlier_counts
      
            
def find_outliers(data:pd.DataFrame, column:str, threshold=2):
    """
    Function to find outliers in a given column of a DataFrame.

    Parameters:
    - data: DataFrame containing the data.
    - column: Name of the column to detect outliers.
    - threshold: limit of standard deviation to look for

    Returns:
    - DataFrame containing only the outliers.
    """
    datacopy= data.copy()
    datacopy=datacopy.sort_values(by=column, ascending=False)
    mean= datacopy[column].mean()
    std = datacopy[column].std()
    outliers = datacopy[(data[column] > mean + threshold*std)]
    data=datacopy.drop(outliers.index)
    return data, outliers



def show_clusters(data, cluster_column, cluster_numbers):
    """
    Function to print the means of the specified clusters side by side with the mean of the entire dataset,
    along with the sizes of the clusters and the dataset.

    Parameters:
    - data: DataFrame containing the data.
    - cluster_column: Name of the column containing the cluster assignments.
    - cluster_numbers: List of cluster numbers to compare against the overall mean.
    """
    # Ensure the cluster column exists in the DataFrame
    if cluster_column not in data.columns:
        print(f"Column '{cluster_column}' not found in the DataFrame.")
        return

    # Calculate the overall mean
    overall_mean = round(data[data.select_dtypes(include=[np.number]).columns].mean(), 2)

    # Initialize the comparison DataFrame with the overall mean
    comparison_df = pd.DataFrame({
        'Overall Mean': overall_mean
     })

    # Calculate and add the means of the specified clusters to the comparison DataFrame
    for cluster_number in cluster_numbers:
        cluster_data = data[data[cluster_column] == cluster_number]
        cluster_mean = round(cluster_data[cluster_data.select_dtypes(include=[np.number]).columns].mean(), 2)
        comparison_df[f'Cluster {cluster_number} Mean'] = cluster_mean

        # Print cluster size
        cluster_size = cluster_data.shape[0]
        print(f"Cluster {cluster_number} size: {cluster_size}")

    # Print overall dataset size
    overall_size = data.shape[0]
    print(f"Overall dataset size: {overall_size}")

    # Print the comparison DataFrame
    return comparison_df