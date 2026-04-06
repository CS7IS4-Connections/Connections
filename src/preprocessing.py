"""
preprocessing.py
----------------
Load VisualNews from Hugging Face (captions + metadata), join with local
article text from the downloaded articles.tar.gz, and produce the base CSV.

Usage:
    # Stratified 60k sample (default):
    python src/preprocessing.py --output data/samples/sample_60k.csv \
        --data-dir articles/articles

    # Dry run (first 500 rows only):
    python src/preprocessing.py --dry-run \
        --output data/samples/sample_dry.csv \
        --data-dir articles/articles

    # Full dataset:
    python src/preprocessing.py --mode full \
        --output data/processed/full.csv \
        --data-dir articles/articles

    # Diagnose article lookup (prints 5 sample rows per source, then exits):
    python src/preprocessing.py --diagnose --data-dir articles/articles

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

# Target rows per category (pooled across sources)
# 6 categories × 15,000 = 90,000 total
TARGET_PER_CATEGORY = 15000

# Only retain these sources — WaPo and USA Today have no matching article JSON
INCLUDE_SOURCES = {"guardian", "bbc"}

# ---------------------------------------------------------------------------
# Source normalisation
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

# ---------------------------------------------------------------------------
# Topic map — rows whose topic is NOT in this dict are DROPPED
# ---------------------------------------------------------------------------

TOPIC_MAP = {
    # Politics & Government
    "politics": "politics_government",
    "uk": "politics_government",
    "us-news": "politics_government",
    "world": "politics_government",
    "australia-news": "politics_government",
    "global-development": "politics_government",
    "uk-news": "politics_government",
    "national_security": "politics_government",
    "international_relations": "politics_government",
    "white_house": "politics_government",
    "government-and-politics": "politics_government",
    "law": "politics_government",
    "u.s.": "politics_government",
    "nation": "politics_government",

    # Sport
    "sport": "sport",
    "football": "sport",
    "cycling": "sport",
    "athletics": "sport",
    "cricket": "sport",
    "tennis": "sport",
    "golf": "sport",
    "rugby": "sport",
    "rugby-union": "sport",
    "rugby-league": "sport",
    "horse-racing": "sport",
    "boxing": "sport",
    "formula1": "sport",
    "swimming": "sport",
    "sports": "sport",
    "baseball": "sport",
    "basketball": "sport",
    "soccer": "sport",
    "nfl": "sport",
    "nba": "sport",
    "mlb": "sport",

    # Business & Economy
    "business": "business_economy",
    "money": "business_economy",
    "economics": "business_economy",
    "economy": "business_economy",
    "finance": "business_economy",
    "financial": "business_economy",
    "sustainable-business": "business_economy",

    # Science & Technology
    "technology": "science_technology",
    "science": "science_technology",
    "environment": "science_technology",
    "media": "science_technology",
    "science_technology": "science_technology",
    "health": "science_technology",
    "energy": "science_technology",
    "tech": "science_technology",
    "nature": "science_technology",
    "climate": "science_technology",

    # Entertainment & Culture
    "film": "entertainment_culture",
    "music": "entertainment_culture",
    "books": "entertainment_culture",
    "artanddesign": "entertainment_culture",
    "fashion": "entertainment_culture",
    "food": "entertainment_culture",
    "travel": "entertainment_culture",
    "lifeandstyle": "entertainment_culture",
    "arts_culture": "entertainment_culture",
    "entertainment": "entertainment_culture",
    "culture": "entertainment_culture",
    "arts": "entertainment_culture",
    "tv-and-radio": "entertainment_culture",
    "stage": "entertainment_culture",
    "games": "entertainment_culture",
    "television": "entertainment_culture",

    # World & Society
    "society": "world_society",
    "education": "world_society",
    "commentisfree": "world_society",
    "inequality": "world_society",
    "law_crime": "world_society",
    "global-development-professionals-network": "world_society",
    "cities": "world_society",
    "religion": "world_society",
    "immigration": "world_society",
    "community": "world_society",
    "news": "world_society",
    "australia": "world_society",
    "conflict_attack": "world_society",
    "disaster_accident": "world_society",
    "weather": "world_society",
    "europe": "world_society",
}


def consolidate_source(raw):
    if not isinstance(raw, str):
        return "unknown"
    return SOURCE_MAP.get(raw.lower().strip(), raw.lower().strip())


def map_topic(raw):
    """Return the mapped category string, or None if the topic is unmapped."""
    if not isinstance(raw, str):
        return None
    return TOPIC_MAP.get(raw.lower().strip())


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


def build_wapo_index(json_dir):
    """
    Index WaPo JSON files by both UUID and 32-char hex IDs.
    WaPo filenames use either UUID format (with dashes) or plain 32-char hex.
    We index both forms so lookups using either representation succeed.
    """
    index = {}
    _hex32 = re.compile(r'^[0-9a-f]{32}$', re.IGNORECASE)
    for fname in os.listdir(json_dir):
        path = os.path.join(json_dir, fname)
        # Try UUID pattern in the filename
        uid = _uuid_in(fname)
        if uid:
            index[uid] = path                          # with dashes, lowercase
            index[uid.replace("-", "")] = path         # without dashes

        # Also index plain 32-char hex stems
        stem = fname
        for suffix in ("-200.json", ".stm-200.json", ".json"):
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                break
        sl = stem.lower()
        if _hex32.match(sl):
            index[sl] = path                           # raw hex
            # Insert dashes to form UUID-like key
            uuid_fmt = f"{sl[0:8]}-{sl[8:12]}-{sl[12:16]}-{sl[16:20]}-{sl[20:32]}"
            index[uuid_fmt] = path

    return index


# Keep old name as alias so existing callers don't break
def build_uuid_index(json_dir):
    return build_wapo_index(json_dir)


def build_guardian_prefix_index(guardian_index, prefix_len=55):
    """
    Secondary index for truncated Guardian article_ids.
    Maps first `prefix_len` chars of each key -> full key (if unambiguous).
    Only includes keys longer than prefix_len (short keys resolve exactly).
    """
    prefix_map = {}
    for key in guardian_index:
        if len(key) > prefix_len:
            prefix = key[:prefix_len]
            prefix_map.setdefault(prefix, []).append(key)
    # Retain only unambiguous prefixes (single match)
    return {p: keys[0] for p, keys in prefix_map.items() if len(keys) == 1}


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
            indexes[src] = build_wapo_index(path)
        print(f"{len(indexes[src]):,} entries")

    # Build Guardian prefix index for truncated article_ids
    if "guardian" in indexes:
        print("  Building guardian prefix index ...", end=" ", flush=True)
        indexes["guardian_prefix"] = build_guardian_prefix_index(indexes["guardian"])
        print(f"{len(indexes['guardian_prefix']):,} entries")

    return json_dirs, indexes


def load_all_pickles(data_dir):
    """
    Scan data_dir recursively for processed_*.p pickle files and load them.
    Returns unified dict: image_id -> {"source_type": str, "article_id": str}
    """
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


def _guardian_variants(article_id):
    """
    Generate candidate lookup keys for a Guardian article_id.
    Tries the raw id, the underscore→slash transform, stripping the
    /items/ prefix, and stripping geographic prefixes (uk/, us/, au/).
    """
    seen = set()
    candidates = []

    def _add(v):
        if v and v not in seen:
            seen.add(v)
            candidates.append(v)

    # 1. Raw article_id as-is
    _add(article_id)

    # 2. Replace underscores with slashes (mirrors build_guardian_index)
    v_slash = article_id.replace("_", "/")
    _add(v_slash)

    # 3. Strip everything before and including /items/
    for v in list(candidates):
        if "/items/" in v:
            _add(v.split("/items/", 1)[1])

    # 4. Strip leading geographic prefix
    for v in list(candidates):
        for prefix in ("uk/", "us/", "au/"):
            if v.startswith(prefix):
                _add(v[len(prefix):])

    return candidates


def lookup_article_text(image_id, pickle_lookup, indexes, diagnose=False):
    """
    Return (raw article text, resolved_key, hit) for a given image_id.
    When diagnose=False the return value is just the raw text string.
    """
    meta = pickle_lookup.get(image_id)
    if not meta:
        if diagnose:
            return "", "", False
        return ""

    source_type = meta["source_type"]
    article_id  = meta["article_id"]
    extractor   = _EXTRACTORS.get(source_type, extract_generic_text)
    filepath    = ""
    resolved    = ""

    # ── Guardian ─────────────────────────────────────────────────────────────
    if source_type == "guardian":
        idx        = indexes.get("guardian", {})
        prefix_idx = indexes.get("guardian_prefix", {})
        for v in _guardian_variants(article_id):
            fp = idx.get(v, "")
            if fp:
                filepath, resolved = fp, v
                break
            # Prefix fallback for truncated article_ids
            prefix = v[:55]
            full_key = prefix_idx.get(prefix, "")
            if full_key:
                fp = idx.get(full_key, "")
                if fp:
                    filepath, resolved = fp, full_key
                    break

    # ── BBC ──────────────────────────────────────────────────────────────────
    elif source_type in ("bbc1", "bbc2"):
        bbc1 = indexes.get("bbc1", {})
        bbc2 = indexes.get("bbc2", {})

        # 1. Trailing number (most common)
        num = _trailing_number(article_id) or ""
        filepath = bbc1.get(num) or bbc2.get(num) or ""
        resolved = num if filepath else ""

        # 2. Raw article_id directly
        if not filepath:
            filepath = bbc1.get(article_id) or bbc2.get(article_id) or ""
            resolved = article_id if filepath else ""

        # 3. Strip http:// / https://
        if not filepath:
            for prefix in ("https://", "http://"):
                if article_id.startswith(prefix):
                    stripped = article_id[len(prefix):]
                    filepath = bbc1.get(stripped) or bbc2.get(stripped) or ""
                    resolved = stripped if filepath else ""
                    break

        if not filepath:
            print(f"BBC miss: image_id={image_id!r}  article_id={article_id!r}",
                  file=sys.stderr)

    # ── USA Today ─────────────────────────────────────────────────────────────
    elif source_type == "usa_today":
        idx = indexes.get("usa_today", {})
        aid = str(article_id)

        # 1. Direct string key
        filepath = idx.get(aid, "")
        resolved = aid if filepath else ""

        # 2. Trailing number
        if not filepath:
            num = _trailing_number(aid) or ""
            filepath = idx.get(num, "")
            resolved = num if filepath else ""

        # 3. Zero-padded to 8 digits
        if not filepath:
            padded = aid.zfill(8)
            filepath = idx.get(padded, "")
            resolved = padded if filepath else ""

    # ── WaPo ─────────────────────────────────────────────────────────────────
    elif source_type == "wapo":
        idx = indexes.get("wapo", {})
        aid_lower = article_id.lower().strip()

        # Build a list of candidate keys to try in order
        wapo_candidates = []

        # 1. UUID extracted from the article_id string (with dashes)
        uid = _uuid_in(article_id)
        if uid:
            wapo_candidates.append(uid)
            wapo_candidates.append(uid.replace("-", ""))  # without dashes

        # 2. Raw lowercased article_id (covers plain 32-char hex)
        wapo_candidates.append(aid_lower)

        # 3. If it's 32 hex chars with no dashes, insert dashes to form UUID
        _hex32 = re.compile(r'^[0-9a-f]{32}$')
        if _hex32.match(aid_lower):
            h = aid_lower
            uuid_fmt = f"{h[0:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"
            wapo_candidates.append(uuid_fmt)

        for candidate in wapo_candidates:
            fp = idx.get(candidate, "")
            if fp:
                filepath, resolved = fp, candidate
                break

    else:
        filepath = ""

    if not filepath or not os.path.exists(filepath):
        if diagnose:
            return "", resolved or article_id, False
        return ""

    text = extractor(filepath)
    if diagnose:
        return text, resolved, bool(text)
    return text


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
# Diagnose mode
# ---------------------------------------------------------------------------

def run_diagnose(dataset, pickle_lookup, indexes):
    """
    Print 5 sample (image_id, article_id, resolved_key, hit/miss) rows per
    source to stderr, then exit.
    """
    source_samples = {}  # source_type -> list of items
    for item in tqdm(dataset.select(range(min(5000, len(dataset)))),
                     desc="Scanning for diagnose samples", leave=False):
        image_id = item.get("id", "")
        meta = pickle_lookup.get(image_id)
        if not meta:
            continue
        src = meta["source_type"]
        if src not in source_samples:
            source_samples[src] = []
        if len(source_samples[src]) < 5:
            source_samples[src].append(image_id)

    print("\n=== DIAGNOSE: 5 sample rows per source ===", file=sys.stderr)
    for src, ids in sorted(source_samples.items()):
        print(f"\n[{src}]", file=sys.stderr)
        hits = 0
        for image_id in ids:
            text, resolved, hit = lookup_article_text(
                image_id, pickle_lookup, indexes, diagnose=True
            )
            article_id = pickle_lookup[image_id]["article_id"]
            status = "HIT " if hit else "MISS"
            print(
                f"  {status}  image_id={image_id!r}  "
                f"article_id={article_id[:60]!r}  "
                f"resolved={resolved[:60]!r}",
                file=sys.stderr
            )
            hits += int(hit)
        print(f"  => {hits}/{len(ids)} hits", file=sys.stderr)
    print("\nDiagnose complete. Exiting.", file=sys.stderr)
    sys.exit(0)


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def process_dataset(dataset, limit, pickle_lookup, indexes):
    """
    Process all (or up to `limit`) rows from the dataset.
    Returns a raw DataFrame before any filtering or sampling.
    Columns: item_id, source, topic_raw, caption, article_lead, caption_len_tokens
    """
    items = dataset if limit is None else dataset.select(range(min(limit, len(dataset))))
    rows = []

    for item in tqdm(items, desc="Processing"):
        image_id  = item.get("id", "")
        topic_raw = str(item.get("topic", "") or "")
        caption   = clean_text(item.get("caption", ""))

        article_raw  = lookup_article_text(image_id, pickle_lookup, indexes)
        article_lead = truncate_tokens(clean_text(article_raw))

        rows.append({
            "item_id":            image_id,
            "source":             consolidate_source(item.get("source", "")),
            "topic_raw":          topic_raw,
            "caption":            caption,
            "article_lead":       article_lead,
            "caption_len_tokens": len(caption.split()),
        })

    return pd.DataFrame(rows, columns=[
        "item_id", "source", "topic_raw", "caption",
        "article_lead", "caption_len_tokens",
    ])


# ---------------------------------------------------------------------------
# Filtering and stratified sampling
# ---------------------------------------------------------------------------

def apply_filters(df):
    """
    Apply the three mandatory filters in order. Returns (filtered_df, counts).
    counts = {"raw": int, "after_caption": int, "after_article": int, "after_topic": int}
    """
    n_raw = len(df)

    # 0. Drop sources with no article coverage (WaPo, USA Today)
    df = df[df["source"].isin(INCLUDE_SOURCES)].copy()
    n_after_source = len(df)

    # 1. Drop rows with short / missing captions
    mask_caption = df["caption"].notna() & (df["caption"].str.strip().str.len() >= 10)
    df = df[mask_caption].copy()
    n_after_caption = len(df)

    # 2. Drop rows with short / missing article_lead (min 100 whitespace-separated tokens)
    mask_article = (
        df["article_lead"].notna()
        & (df["article_lead"].str.strip().str.split().str.len() >= 100)
    )
    df = df[mask_article].copy()
    n_after_article = len(df)

    # 3. Apply TOPIC_MAP — drop unmapped topics, add category column
    df["category"] = df["topic_raw"].apply(map_topic)
    df = df[df["category"].notna()].copy()
    n_after_topic = len(df)

    counts = {
        "raw":            n_raw,
        "after_source":   n_after_source,
        "after_caption":  n_after_caption,
        "after_article":  n_after_article,
        "after_topic":    n_after_topic,
    }
    return df, counts


def stratified_sample(df):
    """
    Sample up to TARGET_PER_CATEGORY rows per category (pooled across sources).
    Within each category, samples proportionally from each source.
    Returns (sampled_df, cell_log).
    """
    sampled, cell_log = [], []

    for category, cat_group in df.groupby("category"):
        available = len(cat_group)
        n_cat = min(available, TARGET_PER_CATEGORY)
        # Sample at category level (preserves natural source proportions)
        cat_sample = cat_group.sample(n=n_cat, random_state=42)
        sampled.append(cat_sample)

        # Log per-source breakdown within this category
        for source, src_group in cat_sample.groupby("source"):
            orig = len(cat_group[cat_group["source"] == source])
            cell_log.append({
                "source":       source,
                "category":     category,
                "available":    orig,
                "sampled":      len(src_group),
                "undersampled": available < TARGET_PER_CATEGORY,
            })

    df_out = pd.concat(sampled).reset_index(drop=True)
    return df_out, cell_log


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Preprocess VisualNews dataset")
    parser.add_argument("--mode",     choices=["stratified", "full"], default="stratified",
                        help="'stratified' = balanced 60k sample (default); 'full' = entire dataset")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Process only the first 500 rows (smoke test)")
    parser.add_argument("--diagnose", action="store_true",
                        help="Print 5 sample lookup rows per source then exit")
    parser.add_argument("--output",   type=str, default="data/samples/sample_90k.csv",
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

    # --- Diagnose mode ---
    if args.diagnose:
        run_diagnose(ds, pickle_lookup, indexes)
        # run_diagnose calls sys.exit()

    # --- Process ---
    if args.dry_run:
        print("\nDRY-RUN mode: processing first 500 rows only.\n")
        limit = 500
    else:
        limit = None  # full dataset for further filtering/sampling

    df = process_dataset(ds, limit, pickle_lookup, indexes)

    # --- Filter ---
    df, counts = apply_filters(df)

    print(f"\n=== Row counts ===")
    print(f"  Raw loaded            : {counts['raw']:>8,}")
    print(f"  After source filter   : {counts['after_source']:>8,}  "
          f"(dropped {counts['raw'] - counts['after_source']:,} — wapo/usatoday)")
    print(f"  After caption filter  : {counts['after_caption']:>8,}  "
          f"(dropped {counts['after_source'] - counts['after_caption']:,})")
    print(f"  After article filter  : {counts['after_article']:>8,}  "
          f"(dropped {counts['after_caption'] - counts['after_article']:,})")
    print(f"  After topic filter    : {counts['after_topic']:>8,}  "
          f"(dropped {counts['after_article'] - counts['after_topic']:,})")

    # --- Stratified sampling (skipped in full mode and dry-run) ---
    if args.mode == "stratified" and not args.dry_run:
        df, cell_log = stratified_sample(df)

        print(f"\n=== Stratified sampling cell log ===")
        print(f"  {'source':<12} {'category':<25} {'available':>10} {'sampled':>8} {'undersampled':>13}")
        print(f"  {'-'*12} {'-'*25} {'-'*10} {'-'*8} {'-'*13}")
        for row in sorted(cell_log, key=lambda r: (r["source"], r["category"])):
            flag = " *" if row["undersampled"] else ""
            print(f"  {row['source']:<12} {row['category']:<25} "
                  f"{row['available']:>10,} {row['sampled']:>8,}{flag}")
        undersampled_count = sum(1 for r in cell_log if r["undersampled"])
        print(f"\n  * = undersampled ({undersampled_count} cells)")
        print(f"  Final sample size: {len(df):,}")
    else:
        # Ensure category column has proper values for full/dry-run modes too
        pass

    # --- Column order for output ---
    output_cols = [
        "item_id", "source", "topic_raw", "category",
        "caption", "article_lead", "caption_len_tokens",
    ]
    df = df[output_cols]

    # --- Save ---
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    df.to_csv(args.output, index=False)

    # --- Summary ---
    filled = (df["article_lead"].str.strip() != "").sum()
    print(f"\n=== Summary ===")
    print(f"Rows written       : {len(df):,}")
    print(f"Output             : {args.output}")
    print(f"article_lead filled: {filled:,} / {len(df):,} ({100*filled/len(df):.1f}%)")
    print(f"\nSource distribution:")
    print(df["source"].value_counts().to_string())
    print(f"\nCategory distribution:")
    print(df["category"].value_counts().to_string())
    print(f"\nEmpty captions     : {(df['caption'].str.strip() == '').sum():,}")
    print("===============\n")


if __name__ == "__main__":
    main()
