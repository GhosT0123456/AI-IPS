import sys
import pandas as pd
from catboost import CatBoostClassifier

def main(model_path, input_parquet, output_parquet=None):
    model = CatBoostClassifier()
    model.load_model(model_path, format='cbm')
    df = pd.read_parquet(input_parquet)
    expected = model.feature_names_
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise SystemExit("Missing required columns for model: " + ", ".join(missing))
    X = df[expected]
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]
    df['prediction'] = preds
    df['prob_pos'] = probs
    if output_parquet:
        df.to_parquet(output_parquet, index=False)
    else:
        print(df[['prediction','prob_pos']].head())

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python run_direct_model.py <model_path> <input_parquet> [output_parquet]"); sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)