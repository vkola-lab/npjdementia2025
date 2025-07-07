import os
import pandas as pd

# Determine repo root
def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    # Paths to filtered data
    train_path = os.path.join(repo_root, 'data', 'ml_data', 'ml_data_filtered.csv')
    test_path = os.path.join(repo_root, 'data', 'ml_data', 'ml_test_data_filtered.csv')

    # Load datasets
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    labels = ['AD', 'nAD', 'VD', 'PRD', 'FTD']

    print("Unique NACCIDs in train set:")
    for label in labels:
        count = train_df[train_df[label] == 1]['NACCID'].nunique()
        print(f"  {label}: {count}")

    print("\nUnique NACCIDs in test set:")
    for label in labels:
        count = test_df[test_df[label] == 1]['NACCID'].nunique()
        print(f"  {label}: {count}")

if __name__ == '__main__':
    main()
