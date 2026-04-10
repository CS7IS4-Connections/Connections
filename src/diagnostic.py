"""
diagnostic.py  —  run this BEFORE touching preprocessing.py
Paste into a file and run: python diagnostic.py --data-dir articles/articles

It will tell you exactly why ~50% of article_lead rows are empty.
"""

import os, pickle, re, json, sys
import argparse
from collections import defaultdict

FWCOLON = "\uf03a"

def _trailing_number(s):
    m = re.search(r"(\d+)[^\d]*$", s)
    return m.group(1) if m else None

def _uuid_in(s):
    m = re.search(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", s, re.IGNORECASE)
    return m.group(0).lower() if m else None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    args = parser.parse_args()
    d = args.data_dir

    print("=" * 60)
    print("1. PICKLE FILES — checking existence and entry counts")
    print("=" * 60)
    pickle_names = [
        "processed_guardian.p",
        "processed_bbc_1.p",
        "processed_bbc_2.p",
        "processed_usa_today.p",
        "processed_washington_post.p",
    ]
    pickles = {}
    for pname in pickle_names:
        path = os.path.join(d, pname)
        if os.path.exists(path):
            with open(path, "rb") as f:
                data = pickle.load(f)
            print(f"  FOUND  {pname:40s}  {len(data):>8,} entries")
            # Sample 3 entries to show structure
            for i, (k, v) in enumerate(list(data.items())[:3]):
                print(f"           sample entry: image_id={k!r}  meta={v}")
            pickles[pname] = data
        else:
            # Search recursively
            found = []
            for root, _, files in os.walk(d):
                if pname in files:
                    found.append(os.path.join(root, pname))
            if found:
                print(f"  FOUND (subdir) {pname:35s} -> {found[0]}")
            else:
                print(f"  MISSING {pname}")

    print()
    print("=" * 60)
    print("2. JSON DIRECTORIES — checking existence and file counts")
    print("=" * 60)
    # Walk and find known dirs
    known = {
        "bbcnews_json":      "bbc1",
        "bbcnews_stm2json":  "bbc2",
        "usa_today_json":    "usa_today",
        "washington_post_json": "wapo",
    }
    found_dirs = {}
    guardian_json_dirs = []
    for root, dirs, files in os.walk(d):
        dirs[:] = [x for x in dirs if x != "old"]
        bname = os.path.basename(root)
        parent = os.path.basename(os.path.dirname(root))
        if bname in known:
            src = known[bname]
            count = len([f for f in os.listdir(root) if f.endswith(".json")])
            print(f"  FOUND  {bname:30s} ({src:10s})  {count:>8,} JSON files  path={root}")
            found_dirs[src] = root
        if bname == "guardian_json" and "guardian" in parent:
            count = len([f for f in os.listdir(root) if f.endswith(".json")])
            print(f"  FOUND  guardian_json                ({count:>8,} JSON files)  path={root}")
            guardian_json_dirs.append((count, root))
            found_dirs["guardian"] = root

    print()
    print("=" * 60)
    print("3. SAMPLE LOOKUPS — tracing 10 items per source through full pipeline")
    print("=" * 60)

    # Build minimal indexes for testing
    def build_guardian_index(json_dir):
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
            index[art_id] = fname
        return index

    def build_number_index(json_dir):
        index = {}
        for fname in os.listdir(json_dir):
            if not fname.endswith(".json"):
                continue
            stem = fname[:fname.rfind("-200.json")] if "-200.json" in fname else fname
            num = _trailing_number(stem)
            if num:
                index[num] = fname
        return index

    def build_uuid_index(json_dir):
        index = {}
        for fname in os.listdir(json_dir):
            uid = _uuid_in(fname)
            if uid:
                index[uid] = fname
        return index

    def build_usatoday_index(json_dir):
        index = {}
        for fname in os.listdir(json_dir):
            if fname.endswith(".json"):
                index[fname[:-5]] = fname
        return index

    indexes = {}
    if "guardian" in found_dirs:
        indexes["guardian"] = build_guardian_index(found_dirs["guardian"])
        print(f"  Guardian index built: {len(indexes['guardian']):,} entries")
    if "bbc1" in found_dirs:
        indexes["bbc1"] = build_number_index(found_dirs["bbc1"])
        print(f"  BBC1 index built:     {len(indexes['bbc1']):,} entries")
    if "bbc2" in found_dirs:
        indexes["bbc2"] = build_number_index(found_dirs["bbc2"])
        print(f"  BBC2 index built:     {len(indexes['bbc2']):,} entries")
    if "usa_today" in found_dirs:
        indexes["usa_today"] = build_usatoday_index(found_dirs["usa_today"])
        print(f"  USAToday index built: {len(indexes['usa_today']):,} entries")
    if "wapo" in found_dirs:
        indexes["wapo"] = build_uuid_index(found_dirs["wapo"])
        print(f"  WaPo index built:     {len(indexes['wapo']):,} entries")

    print()

    # Now trace 10 lookups per source through each pickle
    source_pickle_map = {
        "processed_guardian.p":        ("guardian",  "guardian"),
        "processed_bbc_1.p":           ("bbc1",      "bbc1"),
        "processed_bbc_2.p":           ("bbc2",      "bbc2"),
        "processed_usa_today.p":       ("usa_today", "usa_today"),
        "processed_washington_post.p": ("wapo",      "wapo"),
    }

    for pname, data in pickles.items():
        src_type, idx_key = source_pickle_map[pname]
        print(f"  --- {pname} (source_type={src_type}) ---")
        hits = 0
        misses_no_article_id = 0
        misses_no_index_entry = 0
        misses_no_file = 0

        sample_items = list(data.items())[:200]
        for image_id, meta in sample_items:
            article_id = str(meta.get("article_id", ""))
            if not article_id:
                misses_no_article_id += 1
                continue

            # Reproduce exact lookup logic
            filepath = None
            if src_type == "guardian":
                filepath = indexes.get("guardian", {}).get(article_id)
            elif src_type in ("bbc1", "bbc2"):
                num = _trailing_number(article_id) or ""
                filepath = (indexes.get("bbc1", {}).get(num) or
                            indexes.get("bbc2", {}).get(num))
            elif src_type == "usa_today":
                filepath = indexes.get("usa_today", {}).get(article_id)
                if not filepath:
                    num = _trailing_number(article_id) or ""
                    filepath = indexes.get("usa_today", {}).get(num)
            elif src_type == "wapo":
                uid = _uuid_in(article_id) or article_id.lower()
                filepath = indexes.get("wapo", {}).get(uid)

            if filepath:
                hits += 1
                if hits <= 2:
                    print(f"    HIT:  image_id={image_id!r} article_id={article_id!r} -> {filepath}")
            else:
                if misses_no_index_entry < 3:
                    print(f"    MISS: image_id={image_id!r} article_id={article_id!r} -> no index match")
                    # Show what the index keys look like for comparison
                    idx = indexes.get(idx_key, {})
                    sample_keys = list(idx.keys())[:3]
                    print(f"          index sample keys: {sample_keys}")
                misses_no_index_entry += 1

        total = len(sample_items)
        print(f"    Result: {hits}/{total} hits  |  "
              f"no_article_id={misses_no_article_id}  "
              f"no_index_entry={misses_no_index_entry}  "
              f"checked 200 items")
        print()

if __name__ == "__main__":
    main()
