import os
import sys
import pickle
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool

def load_pipeline(pipeline_path):
    with open(pipeline_path, 'rb') as f:
        return pickle.load(f)

def apply_manual_pipeline(df, state):
    # 1) Feature engineering (same as your Preprocessor.feature_selection)
    eps = 1e-10
    # helper to return a numeric Series (NaN if missing)
    def get_series(col):
        if col in df.columns:
            return pd.to_numeric(df[col], errors='coerce')
        return pd.Series(np.nan, index=df.index)

    sbytes = get_series('sbytes')
    dbytes = get_series('dbytes')
    spkts = get_series('spkts')
    dpkts = get_series('dpkts')
    dur = get_series('dur')
    sloss = get_series('sloss')

    df.loc[:, "Speed of Operations to Speed of Data Bytes"] = np.log1p(sbytes / (dbytes + eps))
    df.loc[:, "Time for a Single Process"] = np.log1p(dur / (spkts + eps))
    df.loc[:, "Ratio of Data Flow"] = np.log1p(dbytes / (sbytes + eps))
    df.loc[:, "Ratio of Packet Flow"] = np.log1p(dpkts / (spkts + eps))
    df.loc[:, "Total Page Errors"] = np.log1p(dur * sloss)
    df.loc[:, "Network Usage"] = np.log1p(sbytes + dbytes)
    df.loc[:, "Network Activity Rate"] = np.log1p(spkts + dpkts)

    # 2) Category mapping (keep only top categories, replace others with '-')
    top_proto = state.get('top_prop_categories') or state.get('top_proto_categories') or []
    top_service = state.get('top_service_categories') or []
    top_state = state.get('top_state_categories') or []

    for col, top in [('proto', top_proto), ('service', top_service), ('state', top_state)]:
        if col in df.columns:
            if top:
                df.loc[:, col] = np.where(df[col].isin(top), df[col], '-')
            # ensure no NaNs in categorical cols
            df.loc[:, col] = df[col].fillna('-')
        else:
            # create column if missing
            df.loc[:, col] = '-'

    # 3) Apply log1p features from pipeline state if available
    for feat in state.get('log_features', []):
        if feat in df.columns:
            df.loc[:, feat] = np.log1p(df[feat].astype(float)).astype('float32')

    # 4) Drop features the pipeline removed during training
    for f in state.get('features_to_drop', []):
        if f in df.columns:
            df = df.drop(columns=[f])

    # 5) Ensure dtypes: categorical_features -> 'category'; numerics -> float
    categorical = state.get('categorical_features', ['proto','service','state'])
    for c in categorical:
        if c in df.columns:
            df.loc[:, c] = df[c].astype('category')
    # convert remaining columns to float where possible
    for col in df.columns:
        if col not in categorical:
            try:
                df.loc[:, col] = df[col].astype(float)
            except Exception:
                # leave non-numeric as-is
                pass

    # 6) Select and order features exactly as training
    selected = state.get('selected_features') or state.get('feature_names') or []
    if not selected:
        # fallback: if model has feature_names_ we'll use later; for now return df
        return df, categorical, selected

    # fill missing selected features with NaN
    for s in selected:
        if s not in df.columns:
            df.loc[:, s] = np.nan

    df = df[selected]
    return df, categorical, selected

def main(pipeline_pkl, model_path, input_parquet, output_parquet=None):
    # load pipeline state
    state = load_pipeline(pipeline_pkl)
    # load model
    model = CatBoostClassifier()
    model.load_model(model_path, format='cbm')

    # load data
    df_raw = pd.read_parquet(input_parquet)

    # ensure numeric columns are numeric (coerce non-numeric -> NaN)
    numeric_cols = [
        'sbytes','dbytes','spkts','dpkts','dur','sloss',
        'flow_byts_s','fwd_pkts_s','bwd_pkts_s',
        'fwd_pkt_len_mean','bwd_pkt_len_mean'
    ]
    for c in numeric_cols:
        if c in df_raw.columns:
            # strip non-numeric characters then coerce
            df_raw.loc[:, c] = pd.to_numeric(df_raw[c].astype(str).str.replace(r'[^0-9\.\-eE]', '', regex=True), errors='coerce')

    # apply pipeline manually
    X, categorical, selected = apply_manual_pipeline(df_raw.copy(), state)

    # if selected_features not in pipeline, fall back to model.feature_names_
    if not selected:
        selected = getattr(model, 'feature_names_', None) or []
        # ensure order and presence
        for s in selected:
            if s not in X.columns:
                X.loc[:, s] = np.nan
        X = X[selected]

    # create CatBoost Pool and predict
    pool = Pool(data=X, cat_features=[c for c in categorical if c in X.columns])
    preds = model.predict(pool)
    try:
        probs = model.predict_proba(pool)
    except Exception:
        probs = None

    # attach results to original frame (aligned by index)
    out = df_raw.copy()
    out['prediction'] = preds
    if probs is not None:
        # if multilabel/proba shape, try to pick positive-class column
        if probs.ndim == 2 and probs.shape[1] > 1:
            out['prob_pos'] = probs[:, 1]
        else:
            out['prob_pos'] = probs.ravel()

    if output_parquet:
        out.to_parquet(output_parquet, index=False)
        print("Wrote predictions to", output_parquet)
    else:
        print(out[['prediction','prob_pos']])

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Usage: python manual_infer.py <pipeline.pkl> <model.cbm> <input.parquet> [output.parquet]")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else None)