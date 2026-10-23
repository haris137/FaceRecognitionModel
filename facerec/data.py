"""Dataset helpers: processed metadata, identity-disjoint splits, verification pairs, subgroups."""
import itertools
import json
import os

import numpy as np
import pandas as pd

from .config import TAG_GROUPS

IDENTITY_ATTRS = ["gender", "age_group", "region", "skin_tone", "source"]


def load_metadata(processed_dir):
    df = pd.read_csv(os.path.join(processed_dir, "images.csv"), dtype={"identity": str})
    df["tags"] = df["tags"].fillna("").astype(str)
    ids_path = os.path.join(processed_dir, "identities.csv")
    if os.path.exists(ids_path):
        ids = pd.read_csv(ids_path, dtype=str).fillna("unknown")
        ids.columns = [c.strip().lower() for c in ids.columns]
        keep = ["identity"] + [c for c in IDENTITY_ATTRS if c in ids.columns and c not in df.columns]
        df = df.merge(ids[keep], on="identity", how="left")
    for c in IDENTITY_ATTRS:
        if c not in df.columns:
            df[c] = "unknown"
        df[c] = df[c].fillna("unknown").astype(str).str.strip().str.lower()
    for g, vocab in TAG_GROUPS.items():
        df[g] = df["tags"].apply(lambda s, v=vocab: any(t in v for t in s.split("|") if t))
    return df


def load_splits(processed_dir):
    with open(os.path.join(processed_dir, "splits.json")) as f:
        return json.load(f)


def subset(df, splits, name, usable_only=False):
    out = df[df["identity"].isin(set(splits[name]))].reset_index(drop=True)
    if usable_only and "usable" in out.columns:
        out = out[out["usable"].astype(bool)].reset_index(drop=True)
    return out


def build_pairs(df, max_genuine_per_id=60, impostor_ratio=1.0, same_gender_frac=0.5, seed=0,
                avoid_same_source=True):
    """Identity-disjoint verification pairs (i, j, label) over rows of df.

    * genuine: pairs of images of the same person (capped per identity so that
      people with many photos do not dominate);
    * impostor: different people; half drawn within the same gender, which is
      the harder and more realistic case.
    * avoid_same_source: frames from the same video are near-duplicates and
      would inflate genuine scores, so they are not paired with each other.
    """
    rng = np.random.default_rng(seed)
    groups = df.groupby("identity").indices
    src = df["source_file"].values if "source_file" in df.columns else None
    genuine = []
    for _, idx in groups.items():
        pairs = [(a, b) for a, b in itertools.combinations(idx, 2)
                 if not (avoid_same_source and src is not None and src[a] == src[b] and str(src[a]).startswith("video:"))]
        if len(pairs) > max_genuine_per_id:
            pairs = [pairs[k] for k in rng.choice(len(pairs), max_genuine_per_id, replace=False)]
        genuine += pairs
    n_imp = int(len(genuine) * impostor_ratio)
    ids = df["identity"].values
    gender = df["gender"].values if "gender" in df.columns else np.array(["unknown"] * len(df))
    by_gender = {g: np.where(gender == g)[0] for g in np.unique(gender)}
    impostor, seen = [], set()
    tries = 0
    while len(impostor) < n_imp and tries < n_imp * 50:
        tries += 1
        a = int(rng.integers(len(df)))
        if rng.random() < same_gender_frac and gender[a] != "unknown" and len(by_gender[gender[a]]) > 1:
            b = int(rng.choice(by_gender[gender[a]]))
        else:
            b = int(rng.integers(len(df)))
        if ids[a] == ids[b]:
            continue
        key = (min(a, b), max(a, b))
        if key in seen:
            continue
        seen.add(key)
        impostor.append(key)
    pairs = np.array([(a, b, 1) for a, b in genuine] + [(a, b, 0) for a, b in impostor], dtype=np.int64)
    return pairs


def pair_groups(df, pairs):
    """Yield (attribute, value, boolean mask over pairs) for the subgroup analysis.

    Identity attributes (gender, age, region, skin tone): a pair belongs to group v
    when BOTH people have value v (so impostors are also within-group).
    Image conditions (mask, head covering, ...): a pair belongs to the group when
    at least one of its two images has the condition; 'clean' = neither has any.
    """
    a, b = pairs[:, 0], pairs[:, 1]
    for attr in ["gender", "age_group", "region", "skin_tone", "source"]:
        if attr not in df.columns:
            continue
        va, vb = df[attr].values[a], df[attr].values[b]
        for v in sorted(set(va) | set(vb)):
            if v == "unknown":
                continue
            yield attr, v, (va == v) & (vb == v)
    any_cond = np.zeros(len(pairs), bool)
    for g in TAG_GROUPS:
        if g not in df.columns:
            continue
        m = df[g].values[a] | df[g].values[b]
        any_cond |= m
        yield "condition", g, m
    yield "condition", "clean (no tagged condition)", ~any_cond
    if "blur" in df.columns:
        lowq = (df["usable"].values[a] == False) | (df["usable"].values[b] == False)  # noqa: E712
        yield "quality", "contains rejected-quality image", lowq
