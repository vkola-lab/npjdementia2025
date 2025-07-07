import pandas as pd


def main():
    """Print mean and standard deviation of left/right hippocampal volumes."""
    # Path to the CSV with a unique row per NACCID
    file_path = "data/ml_data/ml_data_filtered_unique_naccid.csv"

    # Column names for hippocampal volumes
    left_col = "Left-Hippocampus_vol"
    right_col = "Right-Hippocampus_vol"

    # Load the data and keep only the columns we need
    df = pd.read_csv(file_path)

    # subse to AD ==1
    df = df[df["AD"] == 1]

    df = df[[left_col, right_col]]

    # Compute statistics
    left_mean = df[left_col].mean()
    left_std = df[left_col].std()
    right_mean = df[right_col].mean()
    right_std = df[right_col].std()

    # Display results
    print(f"Left Hippocampus: mean = {left_mean:.2f}, std = {left_std:.2f}")
    print(f"Right Hippocampus: mean = {right_mean:.2f}, std = {right_std:.2f}")


if __name__ == "__main__":
    main()
