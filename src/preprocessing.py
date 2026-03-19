"""
preprocessing.py
----------------
Load VisualNews from Hugging Face (captions + metadata), join with local
article text from the downloaded articles.tar.gz, and produce the base CSV.

Usage:
    # With article text (recommended):
    python src/preprocessing.py --mode sample --n 5000 \\
        --output data/samples/sample_5k.csv \\
        --data-dir articles/articles

    # Without article text (captions only):
    python src/preprocessing.py --mode sample --n 5000 \\
        --output data/samples/sample_5k.csv

    # Full dataset:
    python src/preprocessing.py --mode full \\
        --output data/processed/full.csv \\
        --data-dir articles/articles

Article data layout (inside --data-dir):
    processed_guardian.p           -- pickle: image_id -> {article_id, ...}
    processed_bbc_1.p
    processed_bbc_2.p
    processed_usa_today.p
    processed_washington_post.p
    guardian_json_sample/guardian_json/  -- Guardian JSON articles
    bbc_1/bbcnews_json/                  -- BBC JSON articles (set 1)
    bbcnews_stm2json/                    -- BBC JSON articles (set 2, extract bbc_2.tar.gz)
    usa_today_json/                      -- USA Today (extract usa_today_articles.tar.gz)
    washington_post_json/                -- WaPo (extract washington_post_articles.tar.gz)
"""

import argparse
import json
import os
import pickle
import re
import sys
import unicodedata

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Full-width colon (U+F03A) — Windows uses this when tar extracts filenames
# that contain colons (which are illegal on Windows NTFS).
FWCOLON = "\uf03a"

MAX_ARTICLE_TOKENS = 512

# Pickle filenames and the source type each covers
PICKLE_CONFIGS = [
    ("guardian",  "processed_guardian.p"),
    ("bbc1",      "processed_bbc_1.p"),
    ("bbc2",      "processed_bbc_2.p"),
    ("usa_today", "processed_usa_today.p"),
    ("wapo",      "processed_washington_post.p"),
]

# ---------------------------------------------------------------------------
# Source / category normalisation
# ---------------------------------------------------------------------------

SOURCE_MAP = {
    "guardian":        "guardian",
    "the guardian":    "guardian",
    "bbc":             "bbc",
    "bbc news":        "bbc",
    "washington post": "wapo",
    "washington_post": "wapo",
    "washingtonpost":  "wapo",
    "usa today":       "usatoday",
    "usa_today":       "usatoday",
    "usatoday":        "usatoday",
}

CATEGORY_MAP = {
    # Politics & Government
    "politics": "politics_government", "politics_elections": "politics_government",
    "government": "politics_government", "election": "politics_government",
    "parliament": "politics_government", "us politics": "politics_government",
    "uk politics": "politics_government", "us news": "politics_government",
    "us-news": "politics_government", "uk-news": "politics_government",
    "international_relations": "politics_government",
    # Sport
    "sport": "sport", "sports": "sport", "football": "sport", "soccer": "sport",
    "cricket": "sport", "tennis": "sport", "athletics": "sport", "golf": "sport",
    "rugby": "sport", "cycling": "sport", "boxing": "sport", "baseball": "sport",
    "basketball": "sport", "nfl": "sport", "nba": "sport", "mlb": "sport", "nhl": "sport",
    # Business & Economy
    "business": "business_economy", "business_economy": "business_economy",
    "economy": "business_economy", "finance": "business_economy",
    "markets": "business_economy", "money": "business_economy",
    "economics": "business_economy", "financial": "business_economy",
    "sustainable-business": "business_economy", "small-business-network": "business_economy",
    # Science & Technology
    "science": "science_technology", "science_technology": "science_technology",
    "technology": "science_technology", "tech": "science_technology",
    "health": "science_technology", "health_medicine_environment": "science_technology",
    "environment": "science_technology", "climate": "science_technology",
    "climate change": "science_technology", "medical": "science_technology",
    "nature": "science_technology", "sustainability": "science_technology",
    "vital-signs": "science_technology",
    # Entertainment & Culture
    "entertainment": "entertainment_culture", "arts_culture": "entertainment_culture",
    "culture": "entertainment_culture", "culture-professionals-network": "entertainment_culture",
    "arts": "entertainment_culture", "artanddesign": "entertainment_culture",
    "film": "entertainment_culture", "music": "entertainment_culture",
    "television": "entertainment_culture", "tv-and-radio": "entertainment_culture",
    "tv": "entertainment_culture", "books": "entertainment_culture",
    "childrens-books-site": "entertainment_culture", "lifestyle": "entertainment_culture",
    "lifeandstyle": "entertainment_culture", "lifeandhealth": "entertainment_culture",
    "fashion": "entertainment_culture", "travel": "entertainment_culture",
    "food": "entertainment_culture", "stage": "entertainment_culture",
    "media": "entertainment_culture", "media-network": "entertainment_culture",
    # World & Society (catch-all)
    "world": "world_society", "world news": "world_society",
    "society": "world_society", "international": "world_society",
    "news": "world_society", "uk": "world_society",
    "australia": "world_society", "australia-news": "world_society",
    "europe": "world_society", "global": "world_society",
    "global-development": "world_society", "opinion": "world_society",
    "commentisfree": "world_society", "education": "world_society",
    "law": "world_society", "law_crime": "world_society", "crime": "world_society",
    "conflict_attack": "world_society", "disaster_accident": "world_society",
    "religion": "world_society", "community": "world_society", "cities": "world_society",
    "weather": "world_society", "careers": "world_society",
}


def consolidate_source(raw):
    if not isinstance(raw, str):
        return "unknown"
    return SOURCE_MAP.get(raw.lower().strip(), raw.lower().strip())


def consolidate_category(raw):
    if not isinstance(raw, str):
        return "world_society"
    return CATEGORY_MAP.get(raw.lower().strip(), "world_society")


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------

def clean_text(text):
    """Unicode-normalise, strip HTML/XML tags, standardise whitespace."""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&[a-zA-Z]+;", " ", text)
    text = re.sub(r"&#\d+;", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def truncate_tokens(text, max_tokens=MAX_ARTICLE_TOKENS):
    return " ".join(text.split()[:max_tokens])


# ---------------------------------------------------------------------------
# Article index building
# ---------------------------------------------------------------------------

def find_json_dirs(data_dir):
    """
    Walk data_dir and locate known JSON article directories.
    Uses both dir name and parent dir name to identify source.
    Skips the 'old' directory (legacy extraction with different filename encoding).
    When multiple directories match the same source, picks the largest one.
    Returns a dict: source_type -> directory path.
    """
    candidates = {}  # source_type -> list of (count, path)
    for root, dirs, files in os.walk(data_dir):
        # Skip the legacy 'old' directory entirely
        dirs[:] = [d for d in dirs if d != "old"]

        name   = os.path.basename(root)
        parent = os.path.basename(os.path.dirname(root))
        try:
            count = len(os.listdir(root))
        except OSError:
            count = 0

        if name == "bbcnews_json":
            candidates.setdefault("bbc1", []).append((count, root))
        elif name == "bbcnews_stm2json":
            candidates.setdefault("bbc2", []).append((count, root))
        elif name == "guardian_json" and "guardian" in parent:
            candidates.setdefault("guardian", []).append((count, root))
        elif "usa_today" in parent:
            candidates.setdefault("usa_today", []).append((count, root))
        elif "washington_post" in parent:
            candidates.setdefault("wapo", []).append((count, root))

    # Pick the directory with the most files for each source
    return {src: max(paths, key=lambda x: x[0])[1] for src, paths in candidates.items()}


def _trailing_number(s):
    """Extract the last contiguous digit sequence from a string."""
    m = re.search(r"(\d+)[^\d]*$", s)
    return m.group(1) if m else None


def _uuid_in(s):
    """Extract a UUID from a string, if present."""
    m = re.search(
        r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
        s, re.IGNORECASE
    )
    return m.group(0).lower() if m else None


def build_guardian_index(json_dir):
    """
    Index guardian JSON files by article_id.

    Filenames use underscores for both '/' and ':' separators, e.g.:
        http___mobile-apps.guardianapis.com_items_media_from-the-archive-blog_2011_may_24_foo-200.json
    We split on '_items_', take the suffix, strip '-200.json', then replace '_' with '/'.
    This works because hyphens are preserved and only slashes become underscores.
    """
    index = {}
    MARKER = "_items_"
    for fname in os.listdir(json_dir):
        if not fname.endswith("-200.json"):
            continue
        pos = fname.find(MARKER)
        if pos == -1:
            continue
        art_id_raw = fname[pos + len(MARKER): -len("-200.json")]
        art_id = art_id_raw.replace("_", "/")
        index[art_id] = os.path.join(json_dir, fname)
    return index


def build_usatoday_index(json_dir):
    """
    USA Today filenames ARE the article_id: e.g. '10000897.json'.
    """
    index = {}
    for fname in os.listdir(json_dir):
        if fname.endswith(".json"):
            key = fname[:-len(".json")]
            index[key] = os.path.join(json_dir, fname)
    return index


def build_number_index(json_dir):
    """
    Index JSON files by the trailing number in the filename.
    Covers BBC (both sets).
    """
    index = {}
    for fname in os.listdir(json_dir):
        if not fname.endswith(".json"):
            continue
        # strip the -200.json / .stm-200.json suffix first
        stem = fname[: fname.rfind("-200.json")] if "-200.json" in fname else fname
        num = _trailing_number(stem)
        if num:
            index[num] = os.path.join(json_dir, fname)
    return index


def build_uuid_index(json_dir):
    """Index JSON files by UUID (Washington Post)."""
    index = {}
    for fname in os.listdir(json_dir):
        uid = _uuid_in(fname)
        if uid:
            index[uid] = os.path.join(json_dir, fname)
    return index


def build_all_indexes(data_dir):
    """
    Locate JSON directories and build lookup indexes.
    Returns (json_dirs, indexes) where indexes maps source_type -> {key: filepath}.
    """
    json_dirs = find_json_dirs(data_dir)
    print("\nJSON directories found:")
    for src, path in json_dirs.items():
        count = len(os.listdir(path))
        print(f"  {src:12s} -> {path}  ({count:,} files)")
    if not json_dirs:
        print("  (none — article_lead will be empty)")
    print()

    indexes = {}
    for src, path in json_dirs.items():
        print(f"  Building index for {src} ...", end=" ", flush=True)
        if src == "guardian":
            indexes[src] = build_guardian_index(path)
        elif src in ("bbc1", "bbc2"):
            indexes[src] = build_number_index(path)
        elif src == "usa_today":
            indexes[src] = build_usatoday_index(path)
        elif src == "wapo":
            indexes[src] = build_uuid_index(path)
        print(f"{len(indexes[src]):,} entries")

    return json_dirs, indexes


def load_all_pickles(data_dir):
    """
    Scan data_dir recursively for processed_*.p pickle files and load them.
    Returns unified dict: image_id -> {"source_type": str, "article_id": str}
    """
    # Map filename stems to source types
    name_to_source = {
        "processed_guardian":        "guardian",
        "processed_bbc_1":           "bbc1",
        "processed_bbc_2":           "bbc2",
        "processed_usa_today":       "usa_today",
        "processed_washington_post": "wapo",
    }
    unified = {}
    for root, _, files in os.walk(data_dir):
        for fname in files:
            if not fname.endswith(".p"):
                continue
            stem = fname[:-2]
            source_type = name_to_source.get(stem)
            if source_type is None:
                continue
            path = os.path.join(root, fname)
            print(f"  Loading pickle: {fname} ...", end=" ", flush=True)
            with open(path, "rb") as f:
                data = pickle.load(f)
            for image_id, meta in data.items():
                unified[image_id] = {
                    "source_type": source_type,
                    "article_id": str(meta.get("article_id", "")),
                }
            print(f"{len(data):,} items")
    return unified


# ---------------------------------------------------------------------------
# Article text extraction from JSON
# ---------------------------------------------------------------------------

def _read_json_safe(filepath):
    try:
        with open(filepath, encoding="utf-8", errors="replace") as f:
            return json.load(f)
    except Exception:
        return {}


def extract_guardian_text(filepath):
    d = _read_json_safe(filepath)
    return d.get("body") or d.get("standFirst") or ""


def extract_bbc_text(filepath):
    d = _read_json_safe(filepath)
    # body is XML — clean_text() will strip tags
    return d.get("body") or d.get("summary") or ""


def extract_generic_text(filepath):
    d = _read_json_safe(filepath)
    for field in ("body", "text", "content", "article", "bodyText", "summary", "standFirst"):
        val = d.get(field, "")
        if isinstance(val, str) and len(val) > 30:
            return val
    return ""


_EXTRACTORS = {
    "guardian":  extract_guardian_text,
    "bbc1":      extract_bbc_text,
    "bbc2":      extract_bbc_text,
    "usa_today": extract_generic_text,
    "wapo":      extract_generic_text,
}


def lookup_article_text(image_id, pickle_lookup, indexes):
    """
    Return raw article text for a given image_id, or '' if not found.
    """
    meta = pickle_lookup.get(image_id)
    if not meta:
        return ""

    source_type = meta["source_type"]
    article_id  = meta["article_id"]
    extractor   = _EXTRACTORS.get(source_type, extract_generic_text)

    if source_type == "guardian":
        filepath = indexes.get("guardian", {}).get(article_id, "")

    elif source_type in ("bbc1", "bbc2"):
        num = _trailing_number(article_id) or ""
        filepath = (
            indexes.get("bbc1", {}).get(num)
            or indexes.get("bbc2", {}).get(num)
            or ""
        )

    elif source_type == "usa_today":
        # article_id IS the numeric ID
        filepath = indexes.get("usa_today", {}).get(article_id, "")
        if not filepath:
            num = _trailing_number(article_id) or ""
            filepath = indexes.get("usa_today", {}).get(num, "")

    elif source_type == "wapo":
        uid = _uuid_in(article_id) or article_id.lower()
        filepath = indexes.get("wapo", {}).get(uid, "")

    else:
        filepath = ""

    if not filepath or not os.path.exists(filepath):
        return ""

    return extractor(filepath)


# ---------------------------------------------------------------------------
# Dataset inspection
# ---------------------------------------------------------------------------

def inspect_schema(dataset):
    sample = dataset[0]
    print("\n--- Dataset schema (dataset[0]) ---")
    for k, v in sample.items():
        print(f"  {k!r}: {str(v)[:120].replace(chr(10), ' ')!r}")
    print("---\n")
    return sample


def inspect_categories(dataset, n=10000):
    size = min(n, len(dataset))
    sample = dataset[0]
    cat_field = next((f for f in ("topic", "category", "section") if f in sample), None)
    if not cat_field:
        print("No category field detected.")
        return None
    cats = set()
    for item in tqdm(dataset.select(range(size)), desc="Scanning categories", leave=False):
        cats.add(item.get(cat_field, ""))
    print(f"\nUnique '{cat_field}' values ({len(cats)} total):")
    for c in sorted(cats, key=str):
        print(f"  {c!r}")
    print()
    return cat_field


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def process_dataset(dataset, n, pickle_lookup, indexes):
    items = dataset if n is None else dataset.select(range(min(n, len(dataset))))
    rows = []

    for item in tqdm(items, desc="Processing"):
        image_id = item.get("id", "")
        caption_raw = item.get("caption", "")
        caption = clean_text(caption_raw)

        article_raw = lookup_article_text(image_id, pickle_lookup, indexes)
        article = clean_text(article_raw)
        article_lead = truncate_tokens(article)

        rows.append({
            "item_id":            image_id,
            "source":             consolidate_source(item.get("source", "")),
            "category":           consolidate_category(item.get("topic", "")),
            "caption":            caption,
            "article_lead":       article_lead,
            "caption_len_tokens": len(caption.split()),
        })

    return pd.DataFrame(rows, columns=[
        "item_id", "source", "category", "caption", "article_lead", "caption_len_tokens"
    ])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Preprocess VisualNews dataset")
    parser.add_argument("--mode",     choices=["sample", "full"], default="sample")
    parser.add_argument("--n",        type=int, default=5000,
                        help="Sample size (--mode sample only)")
    parser.add_argument("--output",   type=str, required=True,
                        help="Output CSV path")
    parser.add_argument("--data-dir", type=str, default=None,
                        help=(
                            "Root of the extracted articles.tar.gz data "
                            "(the directory that contains processed_guardian.p, "
                            "processed_bbc_1.p, etc. and the JSON subdirectories). "
                            "Omit to skip article text."
                        ))
    args = parser.parse_args()

    # --- Load HuggingFace dataset ---
    print("Loading VisualNews from Hugging Face...")
    ds = load_dataset("twelcone/VisualNews", split="train")
    print(f"Dataset loaded: {len(ds):,} rows\n")

    inspect_schema(ds)
    inspect_categories(ds, n=10000)

    # --- Load article indexes (if data-dir provided) ---
    pickle_lookup = {}
    indexes       = {}

    if args.data_dir:
        data_dir = args.data_dir
        print(f"\nLoading article data from: {data_dir}")
        print("Pickle files:")
        pickle_lookup = load_all_pickles(data_dir)
        print(f"  Total pickle entries: {len(pickle_lookup):,}\n")
        _, indexes = build_all_indexes(data_dir)
    else:
        print(
            "\nWARNING: --data-dir not provided. "
            "article_lead will be empty for all rows.\n"
        )

    # --- Process ---
    n = args.n if args.mode == "sample" else None
    df = process_dataset(ds, n, pickle_lookup, indexes)

    assert len(df) > 0, "Output dataframe is empty"

    # --- Save ---
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    df.to_csv(args.output, index=False)

    # --- Summary ---
    filled = (df["article_lead"] != "").sum()
    print(f"\n=== Summary ===")
    print(f"Rows written       : {len(df):,}")
    print(f"Output             : {args.output}")
    print(f"article_lead filled: {filled:,} / {len(df):,} ({100*filled/len(df):.1f}%)")
    print(f"\nSource distribution:")
    print(df["source"].value_counts().to_string())
    print(f"\nCategory distribution:")
    print(df["category"].value_counts().to_string())
    print(f"\nEmpty captions     : {(df['caption'] == '').sum():,}")
    print("===============\n")


if __name__ == "__main__":
    main()
