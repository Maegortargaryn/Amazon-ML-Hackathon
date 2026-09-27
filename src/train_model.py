import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

# Ensure UTF-8 stdout
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from entity_resolution import (
    dice_similarity,
    extract_city,
    get_blocking_keys,
    jaccard,
    jaro_winkler,
    normalized_address_tokens,
    normalized_name_tokens,
)


def load_training_sample(base_path, n_train=12000, n_val=3000):
    """Load stratified training and validation sets of S1 entities with ground truth."""
    gt_path = base_path / "dataset" / "train" / "train_ground_truth.tsv"
    s1_path = base_path / "dataset" / "train" / "train_source1.tsv"

    print(f"Reading ground truth from {gt_path}...")
    gt_df = pd.read_csv(gt_path, sep="\t", nrows=n_train + n_val + 1000)
    sample_gt = gt_df.head(n_train + n_val).copy()

    s1_needed = set(sample_gt["source1_entity_id"])
    target_needed = set()
    gt_map = {}
    for r in sample_gt.itertuples(index=False):
        matched_str = str(r.matched_entity_ids) if pd.notna(r.matched_entity_ids) else ""
        ids = {x.strip() for x in matched_str.split(",") if x.strip()}
        gt_map[r.source1_entity_id] = ids
        target_needed.update(ids)

    print("Reading Source 1 training records...")
    s1_df = pd.read_csv(s1_path, sep="\t")
    s1_sample = s1_df[s1_df["entity_id"].isin(s1_needed)].copy()
    del s1_df

    all_s1_ids = list(sample_gt["source1_entity_id"])
    train_s1_ids = set(all_s1_ids[:n_train])
    val_s1_ids = set(all_s1_ids[n_train:n_train + n_val])

    train_s1 = s1_sample[s1_sample["entity_id"].isin(train_s1_ids)].copy()
    val_s1 = s1_sample[s1_sample["entity_id"].isin(val_s1_ids)].copy()

    print(f"Loaded {len(train_s1):,} Train S1 and {len(val_s1):,} Val S1 records.")
    print(f"Total true target matches required: {len(target_needed):,}")

    s2_needed = {x for x in target_needed if x.startswith("S2-")}
    s3_needed = {x for x in target_needed if x.startswith("S3-")}

    targets_pool = []
    for file_name, needed in [("train_source2.tsv", s2_needed), ("train_source3.tsv", s3_needed)]:
        path = base_path / "dataset" / "train" / file_name
        read_distractors = 0
        for chunk in pd.read_csv(path, sep="\t", chunksize=150_000):
            found = chunk[chunk["entity_id"].isin(needed)]
            if not found.empty:
                targets_pool.append(found)
            if read_distractors < 25_000:
                distr = chunk[~chunk["entity_id"].isin(needed)].head(25_000 - read_distractors)
                targets_pool.append(distr)
                read_distractors += len(distr)
            needed -= set(found["entity_id"])
            if not needed and read_distractors >= 25_000:
                break

    targets_df = pd.concat(targets_pool, ignore_index=True).drop_duplicates(subset=["entity_id"])
    print(f"Loaded {len(targets_df):,} target records (matches + distractors).")
    return train_s1, val_s1, targets_df, gt_map


def build_candidates_from_df(s1_df, targets_df, max_bucket_size=100):
    s1_ids = list(s1_df["entity_id"])
    s1_countries = list(s1_df["country"])
    s1_names = [normalized_name_tokens(x) for x in s1_df["business_name"]]
    s1_addrs = [normalized_address_tokens(x) for x in s1_df["business_address"]]
    s1_raw_names = list(s1_df["business_name"])
    s1_raw_addrs = list(s1_df["business_address"])

    s1_blocks = defaultdict(list)
    for idx in range(len(s1_ids)):
        for k in get_blocking_keys(s1_countries[idx], s1_names[idx], s1_addrs[idx]):
            s1_blocks[k].append(idx)
    s1_blocks = {k: v for k, v in s1_blocks.items() if len(v) <= max_bucket_size}

    t_ids = list(targets_df["entity_id"])
    t_names = list(targets_df["business_name"])
    t_addrs = list(targets_df["business_address"])
    t_countries = list(targets_df["country"])

    candidate_pairs = []
    for i in range(len(t_ids)):
        t_c = t_countries[i]
        t_n_tok = normalized_name_tokens(t_names[i])
        t_a_tok = normalized_address_tokens(t_addrs[i])
        keys = get_blocking_keys(t_c, t_n_tok, t_a_tok)

        possible_s1 = set()
        for k in keys:
            if k in s1_blocks:
                possible_s1.update(s1_blocks[k])

        for s_idx in possible_s1:
            if s1_countries[s_idx] != t_c:
                continue
            candidate_pairs.append({
                "s1_id": s1_ids[s_idx],
                "target_id": t_ids[i],
                "s1_name": s1_raw_names[s_idx],
                "s1_addr": s1_raw_addrs[s_idx],
                "t_name": t_names[i],
                "t_addr": t_addrs[i],
            })

    return pd.DataFrame(candidate_pairs)


def build_idf_dictionaries(names_list, addrs_list):
    print("Building token IDF weight tables...")
    name_counts = Counter()
    for n in names_list:
        for t in set(normalized_name_tokens(n)):
            name_counts[t] += 1
    N_names = len(names_list)
    idf_name = {t: math.log((N_names + 1) / (c + 1)) + 1 for t, c in name_counts.items()}

    addr_counts = Counter()
    for a in addrs_list:
        for t in set(normalized_address_tokens(a)):
            addr_counts[t] += 1
    N_addrs = len(addrs_list)
    idf_addr = {t: math.log((N_addrs + 1) / (c + 1)) + 1 for t, c in addr_counts.items()}

    print(f"IDF dictionary ready: {len(idf_name):,} name tokens, {len(idf_addr):,} address tokens.")
    return idf_name, idf_addr


def extract_features_vectorized(df, idf_name, idf_addr):
    n_rows = len(df)
    s1_names = df["s1_name"].fillna("").astype(str).tolist()
    t_names = df["t_name"].fillna("").astype(str).tolist()
    s1_addrs = df["s1_addr"].fillna("").astype(str).tolist()
    t_addrs = df["t_addr"].fillna("").astype(str).tolist()

    feats = np.zeros((n_rows, 14), dtype=np.float32)

    for i in range(n_rows):
        s1_n_tok = set(normalized_name_tokens(s1_names[i]))
        t_n_tok = set(normalized_name_tokens(t_names[i]))
        s1_a_tok = set(normalized_address_tokens(s1_addrs[i]))
        t_a_tok = set(normalized_address_tokens(t_addrs[i]))

        nd = dice_similarity(s1_n_tok, t_n_tok)
        nj = jaccard(s1_n_tok, t_n_tok)
        ad = dice_similarity(s1_a_tok, t_a_tok)
        aj = jaccard(s1_a_tok, t_a_tok)

        # Weighted name Dice
        inter_n = s1_n_tok & t_n_tok
        w_nd = 0.0
        if inter_n:
            w_num = sum(idf_name.get(t, 6.0) for t in inter_n) * 2.0
            w_den = sum(idf_name.get(t, 6.0) for t in s1_n_tok) + sum(idf_name.get(t, 6.0) for t in t_n_tok)
            w_nd = w_num / w_den if w_den > 0 else 0.0

        # Weighted address Dice
        inter_a = s1_a_tok & t_a_tok
        w_ad = 0.0
        if inter_a:
            w_num = sum(idf_addr.get(t, 4.0) for t in inter_a) * 2.0
            w_den = sum(idf_addr.get(t, 4.0) for t in s1_a_tok) + sum(idf_addr.get(t, 4.0) for t in t_a_tok)
            w_ad = w_num / w_den if w_den > 0 else 0.0

        jw = jaro_winkler(s1_names[i][:40].lower(), t_names[i][:40].lower())

        s1_digits = {tok for tok in s1_a_tok if tok.isdigit()}
        t_digits = {tok for tok in t_a_tok if tok.isdigit()}
        num_m = 1.0 if (s1_digits and (s1_digits & t_digits)) else 0.0
        num_conf = 1.0 if (s1_digits and t_digits and not (s1_digits & t_digits)) else 0.0

        l1_n, l2_n = len(s1_names[i]), len(t_names[i])
        lr_n = min(l1_n, l2_n) / max(l1_n, l2_n) if max(l1_n, l2_n) > 0 else 1.0

        l1_a, l2_a = len(s1_addrs[i]), len(t_addrs[i])
        lr_a = min(l1_a, l2_a) / max(l1_a, l2_a) if max(l1_a, l2_a) > 0 else 1.0

        c1 = extract_city(s1_addrs[i])
        c2 = extract_city(t_addrs[i])
        cm = 1.0 if (c1 and c1 == c2) else 0.0

        comp = 2.6 * nd + 2.2 * nj + 1.4 * ad + 1.1 * aj + 1.0 + 0.6 * cm

        feats[i] = [nd, w_nd, nj, jw, ad, w_ad, aj, num_m, num_conf, lr_n, lr_a, cm, comp, w_nd * w_ad]

    return feats


def evaluate_macro_f05(val_s1_ids, pred_map, gt_map):
    total_f05 = 0.0
    total_precision = 0.0
    total_recall = 0.0
    n = len(val_s1_ids)

    for s1_id in val_s1_ids:
        actual = gt_map.get(s1_id, set())
        predicted = pred_map.get(s1_id, set())
        tp = len(actual & predicted)
        p = tp / len(predicted) if predicted else (1.0 if not actual else 0.0)
        r = tp / len(actual) if actual else 1.0

        if p + r == 0:
            f05 = 0.0
        else:
            f05 = (1.25 * p * r) / (0.25 * p + r)

        total_f05 += f05
        total_precision += p
        total_recall += r

    return total_f05 / n, total_precision / n, total_recall / n


def train_and_save_model(base_dir=None):
    if base_dir is None:
        base_path = SRC_DIR.parents[2]
    else:
        base_path = Path(base_dir)

    print("=" * 65)
    print(" TRAINING HIGH-PRECISION LIGHTGBM MATCHING MODEL")
    print(f" Workspace: {base_path}")
    print("=" * 65)

    train_s1, val_s1, targets_df, gt_map = load_training_sample(base_path, n_train=12000, n_val=3000)
    val_s1_ids = list(val_s1["entity_id"])

    print("\nGenerating candidate pairs via inverted blocking...")
    t0 = time.time()
    train_pairs_df = build_candidates_from_df(train_s1, targets_df)
    val_pairs_df = build_candidates_from_df(val_s1, targets_df)
    print(f"Generated {len(train_pairs_df):,} Train pairs and {len(val_pairs_df):,} Val pairs in {time.time()-t0:.2f}s.")

    train_labels = np.array([
        1 if row.target_id in gt_map.get(row.s1_id, set()) else 0
        for row in train_pairs_df.itertuples(index=False)
    ], dtype=np.int32)

    val_labels = np.array([
        1 if row.target_id in gt_map.get(row.s1_id, set()) else 0
        for row in val_pairs_df.itertuples(index=False)
    ], dtype=np.int32)

    print(f"Train matches: {train_labels.sum():,} / {len(train_labels):,} ({train_labels.mean()*100:.1f}%)")
    print(f"Val matches:   {val_labels.sum():,} / {len(val_labels):,} ({val_labels.mean()*100:.1f}%)")

    # Build IDF dictionaries
    all_names = train_pairs_df["s1_name"].tolist() + targets_df["business_name"].tolist()
    all_addrs = train_pairs_df["s1_addr"].tolist() + targets_df["business_address"].tolist()
    idf_name, idf_addr = build_idf_dictionaries(all_names, all_addrs)

    # Feature extraction
    print("\nExtracting feature vectors for training...")
    t0 = time.time()
    X_train = extract_features_vectorized(train_pairs_df, idf_name, idf_addr)
    X_val = extract_features_vectorized(val_pairs_df, idf_name, idf_addr)
    print(f"Features extracted in {time.time()-t0:.2f}s (Shape: {X_train.shape}).")

    # Train LightGBM model
    print("\nTraining LightGBM Classifier...")
    model = lgb.LGBMClassifier(
        n_estimators=160,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        random_state=42,
        verbosity=-1,
        n_jobs=-1,
    )
    model.fit(X_train, train_labels)

    # Tune threshold on validation set
    print("Evaluating Macro F0.5 across threshold spectrum on validation split...")
    probs = model.predict_proba(X_val)[:, 1]

    best_f05, best_th, best_p, best_r = 0, 0.5, 0, 0
    for th in np.arange(0.35, 0.85, 0.05):
        pred_map = defaultdict(set)
        for i, row in enumerate(val_pairs_df.itertuples(index=False)):
            if probs[i] >= th:
                pred_map[row.s1_id].add(row.target_id)
        f05, p, r = evaluate_macro_f05(val_s1_ids, pred_map, gt_map)
        print(f"  Threshold {th:.2f} -> Macro F0.5: {f05:.4f} | Precision: {p:.4f} | Recall: {r:.4f}")
        if f05 > best_f05:
            best_f05, best_th, best_p, best_r = f05, th, p, r

    print("=" * 65)
    print(f" OPTIMAL THRESHOLD: {best_th:.2f}")
    print(f" Validation Macro F0.5: {best_f05:.4f} (Precision: {best_p:.4f}, Recall: {best_r:.4f})")
    print("=" * 65)

    # Save artifacts
    models_dir = SRC_DIR.parent / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    artifacts_path = models_dir / "matching_model.joblib"

    artifacts = {
        "model": model,
        "idf_name": idf_name,
        "idf_addr": idf_addr,
        "threshold": float(best_th),
    }
    joblib.dump(artifacts, artifacts_path, compress=3)
    print(f"Model artifacts successfully saved to: {artifacts_path} ({os.path.getsize(artifacts_path) / 1e6:.2f} MB)")
    return artifacts_path


if __name__ == "__main__":
    train_and_save_model()
