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
    
    return fishermen