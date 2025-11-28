#!/usr/bin/env python3
import os
import time
import glob
import sys
import pandas as pd
from catboost import CatBoostClassifier
import psycopg2

conn = psycopg2.connect(
    dbname="ips",
    user="postgres",
    password="achour",
    host="localhost"
)
cursor = conn.cursor()

PARQUET_DIR = "/tmp/parquets"
PROCESSED_DIR = "/tmp/parquets/processed"
OUTPUT_DIR = "/tmp/parquets/output"
CHECK_INTERVAL = 2  # seconds

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_model(path):
    model = CatBoostClassifier()
    model.load_model(path, format='cbm')
    print("[✓] Model loaded:", path)
    return model

def process_file(model, file_path):
    basename = os.path.basename(file_path)
    output_path = os.path.join(OUTPUT_DIR, f"pred_{basename}")
    print(f"\n[+] Processing {basename}")
    
    try:
        df = pd.read_parquet(file_path)

        # Convert categorical columns to string
        cat_columns = ['proto', 'service', 'state']  # update this list based on your model
        for c in cat_columns:
            if c in df.columns:
                df[c] = df[c].fillna("nan").astype(str)

        expected = model.feature_names_
        missing = [c for c in expected if c not in df.columns]
        if missing:
            raise SystemExit("❌ Missing model columns: " + ", ".join(missing))

        X = df[expected]
        preds = model.predict(X)
        probs = model.predict_proba(X)[:, 1]
        
        df["prediction"] = preds
        df["prob_pos"] = probs
        real_pred=df["prediction"].mean()
        if real_pred > 0.5 :
            real_pred=1
            print("IT's an attack")
        else : 
            real_pred=0
            print("Generic traffic")
        # Print result in terminal
        
        
        # Insert predictions into database
        for idx, row in df.iterrows():
            cursor.execute("""
                UPDATE flows_meta
                SET prediction = %s
                WHERE parquet_file =%s
            """, (real_pred,file_path))
        
        conn.commit()

        
        # Save output parquet
        df.to_parquet(output_path, index=False)

        
        # Move processed file
        os.rename(file_path, os.path.join(PROCESSED_DIR, basename))

        
    except Exception as e:
        print(f"[!] Error processing {file_path}: {e}")

def watch_loop(model):
    print("[*] Watching directory for new parquet files...\n")
    while True:
        files = glob.glob(os.path.join(PARQUET_DIR, "*.parquet"))
        if files:
            for f in files:
                process_file(model, f)
        time.sleep(CHECK_INTERVAL)

def main():
    if len(sys.argv) < 2:
        print("Usage: python watch_model.py <model.cbm>")
        sys.exit(1)
    
    model_path = sys.argv[1]
    model = load_model(model_path)
    watch_loop(model)

if __name__ == "__main__":
    main()
