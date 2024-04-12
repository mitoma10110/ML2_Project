def extract_education(name):
    if 'Msc.' in name:
        return 'Msc.'
    elif 'Bsc.' in name:
        return 'Bsc.'
    else:
        return None
    
