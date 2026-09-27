import os
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Ensure UTF-8 stdout
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

STOPWORDS = {
    "and", "the", "of", "for", "inc", "llc", "ltd", "limited", "corp",
    "corporation", "company", "co", "private", "pvt", "services", "service",
    "solutions", "group", "business", "center", "centre", "enterprise", "holdings",
    "trading", "international", "industries", "partners", "systems", "products",
    # French entity terms
    "sa", "sas", "sarl", "sasu", "sci", "eurl", "et", "france", "ste", "societe",
}

GENERIC_ADDR_WORDS = {
    "c", "o", "so", "do", "wo", "care", "plot", "no", "flat", "shop", "h", "house", "survey",
    "khasra", "kheta", "po", "box", "pobox", "suite", "ste", "building", "bldg", "room",
    "rm", "floor", "fl", "nr", "near", "opp", "opposite", "behind", "at", "post", "dist",
    "district", "tehsil", "taluk", "taluka", "road", "rd", "street", "st", "lane", "ln",
    "avenue", "ave", "boulevard", "blvd", "drive", "dr", "highway", "hwy", "court", "ct",
    "nagar", "colony", "marg", "gali", "sector", "sec", "phase", "block",
    # French address terms
    "rue", "bd", "chemin", "route", "allee", "place", "de", "des", "du", "la", "le", "les", "d",
}

ALL_ADDR_STOPWORDS = STOPWORDS | GENERIC_ADDR_WORDS

ADDRESS_ALIASES = {
    "st": "street",
    "street": "street",
    "rd": "road",
    "road": "road",
    "ave": "avenue",
    "avenue": "avenue",
    "blvd": "boulevard",
    "boulevard": "boulevard",
    "dr": "drive",
    "drive": "drive",
    "ln": "lane",
    "lane": "lane",
    "ct": "court",
    "court": "court",
    "pl": "place",
    "place": "place",
    "trl": "trail",
    "trail": "trail",
    "hwy": "highway",
    "highway": "highway",
    "apt": "apartment",
    "apartment": "apartment",
    "unit": "unit",
}

_RE_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_RE_SPACES = re.compile(r"\s+")
_RE_DOMAIN = re.compile(r"\.(com|c0m|in|org|net|co|io|fr)\b", re.IGNORECASE)
_RE_DOMAIN_SUFFIX = re.compile(r"(com|c0m|in|org|net|co|io|fr)$", re.IGNORECASE)


def strip_accents(text):
    if not text:
        return ""
    s = str(text)
    if s.isascii():
        return s
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")


def clean_domain(text):
    """Normalize web URLs and domain names found in business names."""
    if not text:
        return ""
    s = _RE_DOMAIN.sub(" ", str(text))
    s = _RE_DOMAIN_SUFFIX.sub("", s)
    return s


def normalize_text(value):
    if value is None or pd.isna(value):
        return ""
    text = clean_domain(value)
    text = strip_accents(text).lower().strip()
    text = text.replace("&", " and ")
    text = text.replace(".", " ")
    text = _RE_NON_ALNUM.sub(" ", text)
    return _RE_SPACES.sub(" ", text).strip()


_RE_DBA = re.compile(
    r"\b(?:trading as|t/a|dba|d/b/a|doing business as|formerly known as|fka|c/o)\b",
    re.IGNORECASE,
)


def clean_dba(name):
    """Normalize common business DBA / trading-as patterns to retain the true trading name."""
    if not name or pd.isna(name):
        return ""
    parts = _RE_DBA.split(str(name))
    if len(parts) > 1:
        return " ".join(parts)
    return str(name)


GENERIC_NAME_WORDS = {
    "hospital", "clinic", "center", "centre", "church", "temple", "express", "deli",
    "store", "shop", "care", "auto", "motors", "associates", "group", "partners",
    "foundation", "institute", "cardiology", "dental", "medical", "health", "wellness",
    "realty", "properties", "capital", "holdings", "enterprises", "trading", "solutions"
}

_RE_HYPHEN_DIGITS = re.compile(r"(?<=\d)-(?=\d)")


def clean_hyphen_digits(text):
    """Rejoin hyphenated unit numbers: B-8-03 -> B-803, 6-2-101 -> 62101."""
    if not text or pd.isna(text):
        return ""
    return _RE_HYPHEN_DIGITS.sub("", str(text))


def is_indic_script(text):
    """Accurately detect Indic scripts (Devanagari, Telugu, Tamil, Bengali, etc.)."""
    for ch in str(text):
        if 0x0900 <= ord(ch) <= 0x0D7F:
            return True
    return False


def extract_bldg_and_postal(tokens):
    """Separate building/street numbers (<= 4 digits) from postal/PIN codes (5-6 digits)."""
    bldg, postal = set(), set()
    for t in tokens:
        sub = re.findall(r"\d+", t)
        for s in sub:
            clean = s.lstrip("0")
            if not clean:
                continue
            if len(clean) in (5, 6):
                postal.add(clean)
            elif len(clean) <= 4:
                bldg.add(clean)
    return bldg, postal


def get_clean_digits(addr_tokens):
    """Extract numeric building/unit identifiers with leading zeros stripped."""
    digs = set()
    for t in addr_tokens:
        if t.isdigit():
            clean = t.lstrip("0")
            if len(clean) >= 2:
                digs.add(clean)
        else:
            sub = re.findall(r"\d+", t)
            for s in sub:
                clean = s.lstrip("0")
                if len(clean) >= 2:
                    digs.add(clean)
    return digs


try:
    from rapidfuzz.distance import JaroWinkler as _RF_JW
    def jaro_winkler(s1, s2):
        return _RF_JW.similarity(s1, s2)
except ImportError:
    def jaro_winkler(s1, s2):
        """Fast pure-Python Jaro-Winkler string similarity for typo resilience."""
        if s1 == s2:
            return 1.0
        len1, len2 = len(s1), len(s2)
        if len1 == 0 or len2 == 0:
            return 0.0
        max_dist = max(len1, len2) // 2 - 1
        match1 = [False] * len1
        match2 = [False] * len2
        matches = 0
        for i in range(len1):
            start = max(0, i - max_dist)
            end = min(i + max_dist + 1, len2)
            for j in range(start, end):
                if match2[j] or s1[i] != s2[j]:
                    continue
                match1[i] = True
                match2[j] = True
                matches += 1
                break
        if matches == 0:
            return 0.0
        t = 0
        point = 0
        for i in range(len1):
            if not match1[i]:
                continue
            while not match2[point]:
                point += 1
            if s1[i] != s2[point]:
                t += 1
            point += 1
        t /= 2
        jaro = (matches / len1 + matches / len2 + (matches - t) / matches) / 3.0
        prefix = 0
        for i in range(min(4, min(len1, len2))):
            if s1[i] == s2[i]:
                prefix += 1
            else:
                break
        return jaro + prefix * 0.1 * (1.0 - jaro)


def normalized_name_tokens(value):
    text = normalize_text(clean_dba(value))
    if not text:
        return []
    tokens = text.split()
    normalized = []
    for tok in tokens:
        tok = ADDRESS_ALIASES.get(tok, tok)
        if (len(tok) > 1 or tok.isdigit()) and tok not in STOPWORDS:
            normalized.append(tok)
    return normalized


def normalized_address_tokens(value):
    text = normalize_text(value)
    if not text:
        return []
    tokens = text.split()
    cleaned = []
    for tok in tokens:
        tok = ADDRESS_ALIASES.get(tok, tok)
        if (len(tok) > 1 or tok.isdigit()) and tok not in ALL_ADDR_STOPWORDS:
            cleaned.append(tok)
    return cleaned


def jaccard(a, b):
    sa = set(a) if not isinstance(a, set) else a
    sb = set(b) if not isinstance(b, set) else b
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def dice_similarity(a, b):
    sa = set(a) if not isinstance(a, set) else a
    sb = set(b) if not isinstance(b, set) else b
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return 2.0 * len(sa & sb) / (len(sa) + len(sb))


def extract_city(address):
    if not address or pd.isna(address):
        return ""
    text = normalize_text(address)
    if not text:
        return ""
    parts = text.split()
    if len(parts) <= 2:
        return " ".join(parts)
    return " ".join(parts[-2:])


def pair_score(left, right):
    left_country = getattr(left, "country", "")
    right_country = getattr(right, "country", "")
    if left_country and right_country and left_country != right_country:
        return 0.0

    left_name_tokens = normalized_name_tokens(getattr(left, "business_name", ""))
    right_name_tokens = normalized_name_tokens(getattr(right, "business_name", ""))
    left_addr_tokens = normalized_address_tokens(getattr(left, "business_address", ""))
    right_addr_tokens = normalized_address_tokens(getattr(right, "business_address", ""))

    name_seq = dice_similarity(left_name_tokens, right_name_tokens)
    addr_seq = dice_similarity(left_addr_tokens, right_addr_tokens)
    name_jac = jaccard(left_name_tokens, right_name_tokens)
    addr_jac = jaccard(left_addr_tokens, right_addr_tokens)
    same_country = 1.0 if left_country == right_country else 0.0

    left_city = extract_city(getattr(left, "business_address", ""))
    right_city = extract_city(getattr(right, "business_address", ""))
    same_city = 1.0 if left_city and left_city == right_city else 0.0

    score = (
        2.6 * name_seq
        + 2.2 * name_jac
        + 1.4 * addr_seq
        + 1.1 * addr_jac
        + 1.0 * same_country
        + 0.6 * same_city
    )
    return score


def blocking_keys(record):
    country = getattr(record, "country", "")
    name_tokens = normalized_name_tokens(getattr(record, "business_name", ""))
    addr_tokens = normalized_address_tokens(getattr(record, "business_address", ""))
    return get_blocking_keys(country, name_tokens, addr_tokens)


def get_blocking_keys(country, name_tok, addr_tok):
    keys = []
    # 1. Name keys
    if len(name_tok) >= 2:
        keys.append(("n2", country, (name_tok[0], name_tok[1])))
    if len(name_tok) >= 3:
        keys.append(("n3", country, (name_tok[0], name_tok[1], name_tok[2])))
    s = sorted(set(name_tok))
    if len(s) >= 2:
        keys.append(("ns", country, (s[0], s[1])))
    elif len(s) == 1:
        keys.append(("n1", country, s[0]))

    if name_tok:
        squashed = "".join(name_tok)
        if len(squashed) >= 5:
            keys.append(("sq", country, squashed[:8]))

    # 2. Address keys: both raw and words-only (ignoring numbers/prefixes)
    if len(addr_tok) >= 2:
        keys.append(("a2", country, (addr_tok[0], addr_tok[1])))

    addr_words = [t for t in addr_tok if not t.isdigit() and len(t) >= 3]
    if len(addr_words) >= 2:
        keys.append(("aw2", country, (addr_words[0], addr_words[1])))
    if len(addr_words) >= 3:
        keys.append(("aw3", country, (addr_words[0], addr_words[1], addr_words[2])))
        keys.append(("aw_sort", country, tuple(sorted(addr_words[:3]))))

    # Numbers
    nums = [t.lstrip("0") for t in addr_tok if t.isdigit() and len(t.lstrip("0")) >= 2]
    for num in nums:
        if len(num) >= 5:
            keys.append(("num_l", country, num))
    if len(nums) >= 2:
        keys.append(("num_2", country, (nums[0], nums[1])))

    if nums and addr_words:
        keys.append(("num_w", country, (nums[0], addr_words[0])))
        if len(addr_words) >= 2:
            keys.append(("num_w2", country, (nums[0], addr_words[1])))

    return keys


def build_block_index(df, max_bucket_size=100):
    block_index = defaultdict(set)
    for record in df.itertuples(index=False):
        for k in blocking_keys(record):
            block_index[k].add(record.entity_id)
    if max_bucket_size:
        block_index = {k: v for k, v in block_index.items() if len(v) <= max_bucket_size}
    return block_index


def generate_candidates(s1_df, s2s3_df):
    block_index = build_block_index(s2s3_df)
    record_lookup = {r.entity_id: r for r in s2s3_df.itertuples(index=False)}

    results = {}
    for row in s1_df.itertuples(index=False):
        s1_id = row.entity_id
        candidates = set()
        for key in blocking_keys(row):
            candidates.update(block_index.get(key, ()))

        cleaned = []
        for candidate_id in sorted(candidates):
            candidate_row = record_lookup.get(candidate_id)
            if candidate_row is None or candidate_row.country != row.country:
                continue
            if pair_score(row, candidate_row) >= 2.0:
                cleaned.append(candidate_id)
        results[s1_id] = cleaned
    return results


def generate_candidates_chunked(s1_df, target_paths, chunksize=200_000, max_bucket_size=100):
    print(f"Indexing {len(s1_df):,} Source 1 reference records...")
    s1_ids = list(s1_df["entity_id"])
    s1_countries = list(s1_df["country"])
    s1_raw_names = list(s1_df["business_name"].fillna("").astype(str))
    s1_raw_addrs = list(s1_df["business_address"].fillna("").astype(str))

    s1_names = [normalized_name_tokens(x) for x in s1_raw_names]
    s1_addrs = [normalized_address_tokens(clean_hyphen_digits(x)) for x in s1_raw_addrs]
    s1_n_sets = [set(x) for x in s1_names]
    s1_a_sets = [set(x) for x in s1_addrs]
    s1_bldg_postal = [extract_bldg_and_postal(a) for a in s1_addrs]
    s1_compact_names = ["".join(n) for n in s1_names]

    s1_blocks = defaultdict(list)
    for idx in range(len(s1_ids)):
        n_tok = s1_names[idx]
        a_tok = s1_addrs[idx]
        for k in get_blocking_keys(s1_countries[idx], n_tok, a_tok):
            s1_blocks[k].append(idx)

    total_keys = len(s1_blocks)
    s1_blocks = {k: v for k, v in s1_blocks.items() if len(v) <= max_bucket_size}
    print(f"Index built: {len(s1_blocks):,} keys (pruned {total_keys - len(s1_blocks):,} keys with > {max_bucket_size} entries).")

    candidate_map = defaultdict(list)
    match_map = defaultdict(list)

    for path in target_paths:
        file_path = Path(path)
        if not file_path.exists():
            continue

        print(f"Streaming target records from {file_path.name}...")
        processed_file_rows = 0

        for chunk in pd.read_csv(file_path, sep="\t", chunksize=chunksize):
            t_ids = list(chunk["entity_id"])
            t_names_raw = list(chunk["business_name"].fillna("").astype(str))
            t_addrs_raw = list(chunk["business_address"].fillna("").astype(str))
            t_countries = list(chunk["country"])
            chunk_len = len(t_ids)
            processed_file_rows += chunk_len

            t_n_toks = [normalized_name_tokens(x) for x in t_names_raw]
            t_a_toks = [normalized_address_tokens(clean_hyphen_digits(x)) for x in t_addrs_raw]
            t_bldg_postal = [extract_bldg_and_postal(a) for a in t_a_toks]
            t_compact_names = ["".join(n) for n in t_n_toks]

            for i in range(chunk_len):
                t_country = t_countries[i]
                t_n_tok = t_n_toks[i]
                t_a_tok = t_a_toks[i]
                keys = get_blocking_keys(t_country, t_n_tok, t_a_tok)

                possible_s1 = set()
                for k in keys:
                    if k in s1_blocks:
                        possible_s1.update(s1_blocks[k])

                if not possible_s1:
                    continue

                t_id = t_ids[i]
                t_n_set = set(t_n_tok)
                t_a_set = set(t_a_tok)
                t_bldg, t_postal = t_bldg_postal[i]
                t_compact = t_compact_names[i]
                t_raw = t_names_raw[i]
                t_is_nan = (len(t_a_set) == 0)

                for s_idx in possible_s1:
                    s1_id = s1_ids[s_idx]
                    
                    s_n_set = s1_n_sets[s_idx]
                    inter_n = s_n_set & t_n_set
                    if not inter_n and s_n_set and t_n_set:
                        continue # No name overlap at all

                    s1_bldg, s1_postal = s1_bldg_postal[s_idx]

                    bldg_conflict = bool(s1_bldg and t_bldg and not (s1_bldg & t_bldg))
                    postal_conflict = bool(s1_postal and t_postal and not (s1_postal & t_postal))
                    if bldg_conflict or postal_conflict:
                        continue

                    s_a_set = s1_a_sets[s_idx]
                    inter_a = s_a_set & t_a_set
                    a_len = len(s_a_set) + len(t_a_set)
                    ad = (2.0 * len(inter_a) / a_len) if a_len else (1.0 if t_is_nan else 0.0)

                    nd_quick = (2.0 * len(inter_n)) / (len(s_n_set) + len(t_n_set)) if s_n_set else 0.0

                    if len(candidate_map[s1_id]) < 15:
                        if nd_quick >= 0.20 or (ad >= 0.30):
                            candidate_map[s1_id].append(t_id)

                    if len(match_map[s1_id]) >= 10:
                        continue

                    bldg_match = bool(s1_bldg and (s1_bldg & t_bldg))
                    postal_match = bool(s1_postal and (s1_postal & t_postal))
                    s_compact = s1_compact_names[s_idx]
                    domain_match = bool(len(s_compact) >= 5 and len(t_compact) >= 5 and (s_compact in t_compact or t_compact in s_compact))

                    is_match = False

                    # 1. Exact Name match logic
                    if s_compact == t_compact and len(s_compact) >= 4:
                        if bldg_match and ad >= 0.25:
                            is_match = True
                        elif postal_match and ad >= 0.40:
                            is_match = True
                        elif ad >= 0.70:
                            is_match = True
                        elif t_is_nan and len(s_compact) >= 8:
                            is_match = True

                    # 2. Domain Match logic
                    elif domain_match:
                        if bldg_match and ad >= 0.25:
                            is_match = True
                        elif postal_match and ad >= 0.40:
                            is_match = True
                        elif ad >= 0.60:
                            is_match = True

                    # 3. High typo tolerance (Jaro-Winkler)
                    elif nd_quick >= 0.60 or ad >= 0.70:
                        jw = jaro_winkler(s1_raw_names[s_idx][:35].lower(), t_raw[:35].lower())
                        if jw >= 0.92 and ad >= 0.80:
                            is_match = True
                        elif jw >= 0.88 and bldg_match and ad >= 0.50:
                            is_match = True
                        elif nd_quick >= 0.90 and bldg_match and postal_match:
                            is_match = True

                    if is_match:
                        match_map[s1_id].append(t_id)

            print(f"  Processed {processed_file_rows:,} rows from {file_path.name}...")

    for s1_id, m_list in match_map.items():
        c_set = set(candidate_map[s1_id])
        for m_id in m_list:
            if m_id not in c_set:
                candidate_map[s1_id].append(m_id)

    return candidate_map, match_map


def score_matches(results_df, ground_truth_df=None):
    if ground_truth_df is None:
        return None
    gt_map = {}
    for record in ground_truth_df.to_dict(orient="records"):
        s1 = record["source1_entity_id"]
        ids = record["matched_entity_ids"]
        if isinstance(ids, str) and ids.strip():
            gt_map[s1] = {x.strip() for x in ids.split(",") if x.strip()}
        else:
            gt_map[s1] = set()

    total = 0.0
    count = 0
    for row in results_df.itertuples(index=False):
        s1_id = row.source1_entity_id
        actual = gt_map.get(s1_id, set())
        predicted = set()
        if pd.notna(row.matched_entity_ids) and str(row.matched_entity_ids).strip():
            predicted = {x.strip() for x in str(row.matched_entity_ids).split(",") if x.strip()}
        tp = len(actual & predicted)
        precision = tp / max(len(predicted), 1)
        recall = tp / max(len(actual), 1)
        if precision == 0 and recall == 0:
            score = 1.0 if not actual else 0.0
        else:
            score = (1.25 * precision * recall) / (0.25 * precision + recall)
        total += score
        count += 1
    return total / count if count else 0.0


def format_output_rows(s1_df, candidate_map, output_name="matching"):
    rows = []
    col_name = "matched_entity_ids" if output_name == "matching" else "candidate_entity_ids"
    for row in s1_df.itertuples(index=False):
        s1_id = row.entity_id
        candidates = sorted(set(candidate_map.get(s1_id, [])))
        rows.append({
            "source1_entity_id": s1_id,
            col_name: ",".join(candidates),
        })
    return pd.DataFrame(rows)


def run_pipeline(base_dir=None, test_dir=None, output_dir=None, chunksize=200_000):
    if base_dir is None:
        base_path = Path(__file__).resolve().parents[3]
    else:
        base_path = Path(base_dir)

    test_path = Path(test_dir) if test_dir else base_path / "dataset" / "test"
    output_path = Path(output_dir) if output_dir else base_path / "output"
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Reading test reference records from {test_path / 'test_source1.tsv'}...")
    s1_test = pd.read_csv(test_path / "test_source1.tsv", sep="\t")
    print(f"Loaded {len(s1_test):,} reference entities.")

    candidate_map, match_map = generate_candidates_chunked(
        s1_test,
        [test_path / "test_source2.tsv", test_path / "test_source3.tsv"],
        chunksize=chunksize,
    )

    # Write candidate_pairs.tsv directly line-by-line
    candidate_file = output_path / "candidate_pairs.tsv"
    print(f"Writing candidate pairs to {candidate_file}...")
    with open(candidate_file, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in s1_test["entity_id"]:
            cand_ids = sorted(set(candidate_map.get(s1_id, [])))
            f.write(f"{s1_id}\t{','.join(cand_ids)}\n")

    # Write matching_results.tsv directly line-by-line
    matching_file = output_path / "matching_results.tsv"
    print(f"Writing matching results to {matching_file}...")
    with open(matching_file, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in s1_test["entity_id"]:
            m_ids = sorted(set(match_map.get(s1_id, [])))
            f.write(f"{s1_id}\t{','.join(m_ids)}\n")

    print("Successfully generated:")
    print(f"  - {matching_file}")
    print(f"  - {candidate_file}")


if __name__ == "__main__":
    run_pipeline()

