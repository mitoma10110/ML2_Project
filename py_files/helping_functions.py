import numpy as np
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
def show_outliers(data, threshold=2):
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
      
            
def find_outliers(data, column, threshold=2):
    """
    Function to find outliers in a given column of a DataFrame.

    Parameters:
    - data: DataFrame containing the data.
    - column: Name of the column to detect outliers.
    - threshold: 

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
