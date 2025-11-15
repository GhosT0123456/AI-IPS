import sys
import pandas as pd
from pipeline_state.pkl import Preprocessor  # adjust import path if needed

def main(model_path, input_parquet, output_parquet=None):
    pre = Preprocessor(model_path)
    df = pd.read_parquet(input_parquet)
    df_pre = pre.preprocess(df)
    pool = pre.create_pool(df_pre)
    preds = pre.model.predict(pool)
    probs = pre.model.predict_proba(pool)[:, 1]  # positive-class prob (binary)
    df['prediction'] = preds
    df['prob_pos'] = probs
    if output_parquet:
        df.to_parquet(output_parquet, index=False)
    else:
        print(df[['prediction','prob_pos']].head())

if __name__ == '__main__':
    # example: python d:\TestML\run_with_preprocessor.py d:\TestML\detection_model_91_F1_V1.cbm d:\TestML\capture_unsw.parquet d:\TestML\capture_preds.parquet
    if len(sys.argv) < 3:
        print("Usage: python run_with_preprocessor.py <model_path> <input_parquet> [output_parquet]"); sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)