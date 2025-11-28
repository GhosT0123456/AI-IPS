#!/usr/bin/env python3
import os
import time
import glob
import pickle
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool

pipeline_path = "pipeline_state.pkl"
MODEL_CBM = "detection_model_catboost.cbm"
CHECK_INTERVAL = 2  # seconds
PARQUET_DIR = "/tmp/parquets"
PROCESSED_DIR = "/tmp/parquets/processed"
OUTPUT_DIR = "/tmp/parquets/output"

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load pipeline
with open(pipeline_path, 'rb') as f:
    pipeline_state = pickle.load(f)

# Load model
model = CatBoostClassifier()
model.load_model(MODEL_CBM, format='cbm')


def apply_pipeline(df, state):
    eps = 1e-10
    def get_series(col):
        return pd.to_numeric(df[col], errors='coerce') if col in df.columns else pd.Series(np.nan, index=df.index)

    sbytes = get_series('sbytes')
    dbytes = get_series('dbytes')
    spkts = get_series('spkts')
    dpkts = get_series('dpkts')
    dur = get_series('dur')
    sloss = get_series('sloss')

    df["Speed of Operations to Speed of Data Bytes"] = np.log1p(sbytes / (dbytes + eps))
    df["Time for a Single Process"] = np.log1p(dur / (spkts + eps))
    df["Ratio of Data Flow"] = np.log1p(dbytes / (sbytes + eps))
    df["Ratio of Packet Flow"] = np.log1p(dpkts / (spkts + eps))
    df["Total Page Errors"] = np.log1p(dur * sloss)
    df["Network Usage"] = np.log1p(sbytes + dbytes)
    df["Network Activity Rate"] = np.log1p(spkts + dpkts)

    # Category mapping
    for col, top in [('proto', state.get('top_proto_categories', [])),
                     ('service', state.get('top_service_categories', [])),
                     ('state', state.get('top_state_categories', []))]:
        if col in df.columns:
            if top:
                df[col] = np.where(df[col].isin(top), df[col], '-')
            df[col] = df[col].fillna('-')
        else:
            df[col] = '-'

    # Log features
    for feat in state.get('log_features', []):
        if feat in df.columns:
            df[feat] = np.log1p(df[feat].astype(float))

    # Drop unused features
    for f in state.get('features_to_drop', []):
        if f in df.columns:
            df = df.drop(columns=[f])

    # Ensure categorical types
    categorical = state.get('categorical_features', ['proto','service','state'])
    for c in categorical:
        if c in df.columns:
            df[c] = df[c].astype('category')

    # Select only features used in training
    selected = state.get('selected_features') or state.get('feature_names') or []
    for s in selected:
        if s not in df.columns:
            df[s] = np.nan
    df = df[selected] if selected else df

    return df, categorical, selected


def process_file(file_path):
    basename = os.path.basename(file_path)
    output_file = os.path.join(OUTPUT_DIR, f"pred_{basename}")

    print(f"[+] Processing {file_path} -> {output_file}")
    try:
        df_raw = pd.read_parquet(file_path)
        X, categorical, selected = apply_pipeline(df_raw.copy(), pipeline_state)

        pool = Pool(data=X, cat_features=[c for c in categorical if c in X.columns])
        preds = model.predict(pool)
        try:
            probs = model.predict_proba(pool)
        except Exception:
            probs = None

        df_out = df_raw.copy()
        df_out['prediction'] = preds
        if probs is not None:
            df_out['prob_pos'] = probs[:, 1] if probs.ndim == 2 and probs.shape[1] > 1 else probs.ravel()

        # Print results to terminal
        print(df_out[['prediction', 'prob_pos']])

        # Save parquet
        df_out.to_parquet(output_file, index=False)
        print(f"[✓] Saved output to {output_file}")

        # Move processed input
        os.rename(file_path, os.path.join(PROCESSED_DIR, basename))

    except Exception as e:
        print(f"[!] Failed to process {file_path}: {e}")


def main_loop():
    print("[*] Watching directory for new parquets...")
    while True:
        parquet_files = glob.glob(os.path.join(PARQUET_DIR, "*.parquet"))
        for file_path in parquet_files:
            process_file(file_path)
        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    main_loop()

