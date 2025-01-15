import os
import pandas as pd
import numpy as np

# Define input and output directories
input_base_dir = os.path.expanduser('../data/input')
output_base_dir = os.path.expanduser('../data/output')

# Define valid ranges for columns
valid_ranges = {
    "scan": (4, 129),
    "begin_time": (56605.38856481481, 58584.34300925926),
    "end_time": (56605.39115740741, 58584.345601851855),
    "spectral_window": 0,
    "channel": (0, 129)
}

# Function to clean a single CSV file
def clean_csv(file_path, output_path):
    # try:
        # Step 1: Read the data
        df = pd.read_csv(file_path)

        # Convert all columns to numeric where applicable
        for col in df.columns:
            if col in valid_ranges.keys() or col in ["max", "mean", "median", "min", "rms", "stddev", "var"]:
                df[col] = pd.to_numeric(df[col], errors='coerce')


        # Step 2: Identify reference combinations (correct combinations of scan, begin and end)
        reference_df = df[(df["scan"].between(valid_ranges["scan"][0], valid_ranges["scan"][1])) &
                          (df["begin_time"].between(valid_ranges["begin_time"][0], valid_ranges["begin_time"][1])) &
                          (df["end_time"].between(valid_ranges["end_time"][0], valid_ranges["end_time"][1]))]

        # Group by unique combinations of scan, begin_time, and end_time
        reference_combinations = reference_df.groupby(["scan", "begin_time", "end_time"]).size()

        # Keep only combinations that occur more than 10 times (assuming that there is lot more clean data than bad data)
        reference_combinations = reference_combinations[reference_combinations > 10]
        reference_combinations = reference_combinations.reset_index()[["scan", "begin_time", "end_time"]]

        # Step 3: Fix rows based on reference combinations
        for idx, row in df.iterrows():
            row_values = (row["scan"], row["begin_time"], row["end_time"])

            # If row values match a reference combination, skip
            if any((row_values == tuple(ref) for ref in reference_combinations.to_records(index=False))):
                continue

            # Find the closest reference combination
            best_match = None
            max_matches = 0
            for _, ref in reference_combinations.iterrows(): # calculate how many out of three values match with ref. Even if one matches, we use it to fix the rest
                matches = sum([row["scan"] == ref["scan"],
                               row["begin_time"] == ref["begin_time"],
                               row["end_time"] == ref["end_time"]])
                if matches > max_matches:
                    max_matches = matches
                    best_match = ref

            # Update row with the best match values
            if best_match is not None:
                if row["scan"] != best_match["scan"]:
                    df.at[idx, "scan"] = best_match["scan"]
                if row["begin_time"] != best_match["begin_time"]:
                    df.at[idx, "begin_time"] = best_match["begin_time"]
                if row["end_time"] != best_match["end_time"]:
                    df.at[idx, "end_time"] = best_match["end_time"]


        # Step 3: Drop rows where scan, begin_time, or end_time could not be fixed
        df = df.dropna(subset=["scan", "begin_time", "end_time"])

        # Step 4: Remove duplicate rows within the same scan and channel
        df = df.drop_duplicates(subset=["scan", "channel"])

        # Step 5: Ensure statistical relationships (max > mean > median > min)
        df = df[(df["max"] > df["mean"]) & (df["max"] > df["median"]) & (df["mean"] > df["median"]) & (df["median"] > df["min"])]

        # Step 6: Ensure positive values for rms, stddev, and var
        df = df[(df["rms"] > 0) & (df["stddev"] > 0) & (df["var"] > 0)]

        # Step 7: Ensure temporal relationship (end_time > begin_time)
        df = df[df["end_time"] > df["begin_time"]]

        # Step 8: Force spectral_window to 0
        df["spectral_window"] = 0

        # Step 9: Ensure valid ranges for all columns
        df = df[(df["scan"] >= valid_ranges["scan"][0]) & (df["scan"] <= valid_ranges["scan"][1])]
        df = df[(df["begin_time"] >= valid_ranges["begin_time"][0]) & (df["begin_time"] <= valid_ranges["begin_time"][1])]
        df = df[(df["end_time"] >= valid_ranges["end_time"][0]) & (df["end_time"] <= valid_ranges["end_time"][1])]
        df = df[df["spectral_window"] == valid_ranges["spectral_window"]]
        df = df[(df["channel"] >= valid_ranges["channel"][0]) & (df["channel"] <= valid_ranges["channel"][1])]

        # Step 10: Sort by scan and then channel
        df = df.sort_values(by=["scan", "channel"])

        # Step 11: Save the cleaned data to the output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cleaned_csv_path = output_path.replace(".csv", "_cleaned.csv")
        df.to_csv(cleaned_csv_path, index=False)

        # Step 12: Save as numpy file
        cleaned_numpy_path = output_path.replace(".csv", ".npy")
        np.save(cleaned_numpy_path, df.values)

        print(f"Successfully cleaned and saved: {cleaned_csv_path} and {cleaned_numpy_path}")

    # except Exception as e:
    #     print(f"Error cleaning file {file_path}: {e}")

# Process all files in the input directory
for year_dir in os.listdir(input_base_dir):
    input_year_path = os.path.join(input_base_dir, year_dir)
    output_year_path = os.path.join(output_base_dir, year_dir)

    if os.path.isdir(input_year_path):
        for file_name in os.listdir(input_year_path):
            if file_name.endswith('.csv'):
                input_file_path = os.path.join(input_year_path, file_name)
                output_file_path = os.path.join(output_year_path, file_name)

                # Clean the CSV file
                clean_csv(input_file_path, output_file_path)

print("Data cleansing process completed.")
