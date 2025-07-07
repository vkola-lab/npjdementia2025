import os
import json

def write_clean_region_names():
    # Determine project root and data directories
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    data_dir = os.path.join(repo_root, 'data')
    results_dir = os.path.join(data_dir, 'results_data')
    os.makedirs(results_dir, exist_ok=True)

    # Load feature configuration and name mapping
    config_file = os.path.join(data_dir, 'feature_config.json')
    names_map_file = os.path.join(data_dir, 'feature_names_map.json')
    with open(config_file) as f:
        config = json.load(f)
    with open(names_map_file) as f:
        names_map = json.load(f)

    # Load lobe mapping and group regions by lobe
    lobe_map_file = os.path.join(data_dir, 'lobe_mapping.json')
    with open(lobe_map_file) as f:
        lobe_map = json.load(f)
    region_keys = config.get('region_names', [])
    # Build clean name lists per lobe
    lobes_clean = {}
    for lobe, keys in lobe_map.items():
        clean_list = [names_map.get(k, k) for k in keys if k in region_keys]
        if clean_list:
            lobes_clean[lobe] = clean_list

    # Write one line per lobe, comma-separated clean region names
    output_file = os.path.join(results_dir, 'region_names.txt')
    with open(output_file, 'w') as f:
        for lobe, names in lobes_clean.items():
            f.write(f"{lobe}: {', '.join(names)}\n")

    print(f"Wrote {sum(len(v) for v in lobes_clean.values())} region names across {len(lobes_clean)} lobes to {output_file}")

if __name__ == '__main__':
    write_clean_region_names()