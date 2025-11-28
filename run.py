import os
import time
import glob
import subprocess
import pandas as pd

# Paths
PARQUET_DIR = "/tmp/parquets"
PROCESSED_DIR = "/tmp/parquets/processed"
OUTPUT_DIR = "/tmp/parquets/output"
PIPELINE_PKL = "pipeline_state.pkl"
MODEL_CBM = "detection_model_catboost.cbm"

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

CHECK_INTERVAL = 2  # seconds

while True:
    parquet_files = glob.glob(os.path.join(PARQUET_DIR, "*.parquet"))

    for file_path in parquet_files:
        basename = os.path.basename(file_path)
        output_file = os.path.join(OUTPUT_DIR, f"pred_{basename}")

        print(f"[+] Processing {file_path} -> {output_file}")

        # Call your manual_infer.py main function
        try:
            # Note: modify manual_infer.py main to return the DataFrame
            subprocess.run([
            "python3", "manual_infer.py",
            PIPELINE_PKL,
            MODEL_CBM,
            f,
            output_file
                        ])

            # Optionally, print the saved parquet to terminal
            df_out = pd.read_parquet(output_file)
            print(df_out[['prediction', 'prob_pos']])

        except Exception as e:
            print(f"[!] Error processing {file_path}: {e}")

        # Move processed parquet
        os.rename(file_path, os.path.join(PROCESSED_DIR, basename))

    time.sleep(CHECK_INTERVAL)

