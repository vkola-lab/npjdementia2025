import json

def load_feature_configs(
    json_path: str
) -> dict:
    """
    Load feature configurations from a JSON file.
    
    Parameters:
    -----------
    json_path : str
        Path to the JSON configuration file
        
    Returns:
    --------
    dict
        Dictionary containing feature configurations
        the dict has keys:
        - 'vol_features': List of volumetric features (just t1)
        - 'wmh_features': List of white matter hyperintensity features (just t2/flair) (includes the total wmh burden)
        - 'imag_features': List of imaging features (vol + wmh)
        - 'demo_features': List of demographic features
        - 'all_features': List of all features (vol + wmh + demo)
    """
    with open(json_path, 'r') as f:
        config = json.load(f)
    
    # Generate derived feature lists
    assert 'region_names' in config, "Configuration must contain 'region_names' key"
    # Create imaging features based on region names
    vol_features = []
    wmh_features = []
    
    for region in config['region_names']:
        # Add volume features
        vol_feature = f"{region}_vol"
        vol_features.append(vol_feature)
        
        # Add WMH features
        wmh_feature = f"{region}_wmh"
        wmh_features.append(wmh_feature)
        
    # Add total WMH burden
    assert 'total_wm_burden' in config, "Configuration must contain 'total_wm_burden' key"
    assert isinstance(config['total_wm_burden'], list) and len(config['total_wm_burden']) == 1, f"Total WMH burden must be list with one element, got {config['total_wm_burden']}"
    wmh_features.append(config['total_wm_burden'][0])

    imag_features = vol_features + wmh_features
    demo_features = config.get('demo_features', [])
    all_features = imag_features + demo_features

    # Update the config dictionary with derived features
    config['vol_features'] = vol_features
    config['wmh_features'] = wmh_features
    config['imag_features'] = imag_features
    config['demo_features'] = demo_features
    config['all_features'] = all_features
    
    return config