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
    datacopy= data.deepcopy()
    datacopy=datacopy.sort_values(by=column, ascending=False)
    mean= datacopy[column].mean()
    std = datacopy[column].std()
    outliers = datacopy[(data[column] > mean + threshold*std)]
    data=datacopy.drop(outliers.index)
    return data, outliers
