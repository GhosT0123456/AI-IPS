import os
import sys
import pandas as pd
import numpy as np
import time
# Mapping decisions (based on provided header)
# - sbytes <- totlen_fwd_pkts (total forward bytes)
# - dbytes <- totlen_bwd_pkts (total backward bytes)
# - spkts  <- tot_fwd_pkts
# - dpkts  <- tot_bwd_pkts
# - dur    <- flow_duration
# - proto  <- protocol
# - smean  <- fwd_pkt_len_mean
# - dmean  <- bwd_pkt_len_mean
# - sinpkt <- fwd_pkts_s (forward pkts/sec)
# - dinpkt <- bwd_pkts_s
# - flow_byts_s provided -> split between sload/dload proportionally by spkts/dpkts if possible
# - service/state default to '-'
input_dir="/tmp/parquets"
output_dir="/tmp/final_parquets"
os.makedirs(output_dir,exist_ok=True)


def convert(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    # basic direct mappings (use .get to avoid KeyError)
    out['src_ip'] = df.get('src_ip')
    out['dst_ip'] = df.get('dst_ip')
    out['proto'] = df.get('protocol').fillna('-') if 'protocol' in df.columns else '-'
    out['service'] = '-'
    out['state'] = '-'

    # core numeric features
    out['sbytes'] = df.get('totlen_fwd_pkts') if 'totlen_fwd_pkts' in df.columns else df.get('subflow_fwd_byts')
    out['dbytes'] = df.get('totlen_bwd_pkts') if 'totlen_bwd_pkts' in df.columns else df.get('subflow_bwd_byts')
    out['spkts']  = df.get('tot_fwd_pkts')  if 'tot_fwd_pkts' in df.columns else df.get('fwd_pkts_s')  # prefer totals
    out['dpkts']  = df.get('tot_bwd_pkts')  if 'tot_bwd_pkts' in df.columns else df.get('bwd_pkts_s')
    out['dur']    = df.get('flow_duration') if 'flow_duration' in df.columns else df.get('flow_duration', np.nan)

    # fallback replace missing with NaN
    for c in ['sbytes','dbytes','spkts','dpkts','dur']:
        if c not in out.columns:
            out[c] = np.nan

    # mean packet lengths
    out['smean'] = df.get('fwd_pkt_len_mean')
    out['dmean'] = df.get('bwd_pkt_len_mean')
    # per-direction pkt/sec approximations
    out['sinpkt'] = df.get('fwd_pkts_s') if 'fwd_pkts_s' in df.columns else (out['spkts'] / out['dur']).replace([np.inf, -np.inf], np.nan)
    out['dinpkt'] = df.get('bwd_pkts_s') if 'bwd_pkts_s' in df.columns else (out['dpkts'] / out['dur']).replace([np.inf, -np.inf], np.nan)

    # sload/dload: split flow_byts_s proportionally by packet counts where possible
    flow_byts_s = df.get('flow_byts_s')
    if flow_byts_s is not None:
        denom = (out['spkts'].fillna(0) + out['dpkts'].fillna(0)).replace(0, np.nan)
        prop_s = (out['spkts'].fillna(0) / denom).fillna(0.5)  # if denom is NA -> split 50/50
        prop_d = 1.0 - prop_s
        out['sload'] = flow_byts_s * prop_s
        out['dload'] = flow_byts_s * prop_d
    else:
        # fallback: approximate using bytes/duration per direction
        out['sload'] = (out['sbytes'] / out['dur']).replace([np.inf, -np.inf], np.nan)
        out['dload'] = (out['dbytes'] / out['dur']).replace([np.inf, -np.inf], np.nan)

    # jitter not present in cicflowmeter export -> set NaN
    out['sjit'] = np.nan
    out['djit'] = np.nan

    # keep other useful features if present (packet-size avg etc.)
    for col in ['pkt_len_mean','pkt_size_avg','pkt_len_std','pkt_len_var','fwd_pkt_len_std','bwd_pkt_len_std']:
        if col in df.columns:
            out[col] = df[col]

    # ensure correct column order: numeric features then categorical at end
    cats = ['proto','service','state']
    cols = [c for c in out.columns if c not in cats] + [c for c in cats if c in out.columns]
    return out[cols]

def main():

    print(f"Monitoring {input_dir} for parquet files...")
    while True:
        for filename in os.listdir(input_dir):
            if not filename.endswith(".parquet"):
                continue

            parquet_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            try:
                print(f"Processing {filename} ...")
                df = pd.read_parquet(parquet_path)

                converted = convert(df)

                essentials = ['sbytes','dbytes','spkts','dpkts','dur','proto']
                missing = [c for c in essentials if converted[c].isna().all()]
                if missing:
                    print("WARNING: missing essential columns:", missing)

                converted.to_parquet(output_path, index=False)
                print(f"Saved → {output_path}")

                # DELETE the processed input
                os.remove(parquet_path)
                print(f"Deleted input file {parquet_path}")

            except Exception as e:
                print(f"Error processing {filename}: {e}")

        # small sleep to avoid busy looping
        time.sleep(0.5)
if __name__ == '__main__':
    
    main()
