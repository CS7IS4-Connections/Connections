# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_trf   # required for all NLP scripts
```

## Commands

```bash
# Preprocessing — build the 5k sample CSV
python src/preprocessing.py --mode sample --n 5000 --output data/samples/sample_5k.csv

# Preprocessing — full dataset
python src/preprocessing.py --mode full --output data/processed/full.csv

# Cleaning — drop rows with empty article_lead
python src/cleaning.py \
    --input  data/samples/sample_5k.csv \
    --output data/samples/sample_data_clean.csv

# Run full pipeline on the cleaned sample (recommended entry point)
python run_pipeline.py \
    --input  data/samples/sample_data_clean.csv \
    --output results/sample_results.csv

# Run individual pipeline stages independently
python src/structural_features.py --input data/samples/sample_data_clean.csv \
    --output results/structural.csv
python src/similarity.py     --input results/structural.csv  --output results/similarity.csv
python src/entity_alignment.py --input results/similarity.csv  --output results/entity.csv
python src/caption_classifier.py --input results/entity.csv   --output results/final.csv
```

There are no automated tests or a linter configured.

## Architecture

The project is a **two-stage NLP pipeline** for analysing caption-article textual similarity using the [VisualNews dataset](https://huggingface.co/datasets/twelcone/VisualNews).

### Stage 1 — `src/preprocessing.py`

Loads VisualNews directly from Hugging Face (`twelcone/VisualNews`). No manual download is needed for the caption data, but full article text requires a separately obtained tar.gz archive (4+ GB, excluded from git).

Key transformations:
- Cleans text: Unicode NFKC normalisation, HTML/XML tag removal, HTML entity decoding, whitespace standardisation.
- Truncates articles to the first 512 whitespace-separated tokens.
- Maps 30+ raw category strings into 5 standardised categories: `politics_government`, `sport`, `business_economy`, `science_technology`, `entertainment_culture`, `world_society`.
- Handles Windows NTFS full-width colon encoding in article filenames.
- Looks up articles via multiple strategies: UUID index, pickle metadata, JSON directory discovery. Different news outlets (Guardian, BBC, USA Today, Washington Post) use different JSON field names.

Output CSV columns: `item_id`, `source`, `category`, `caption`, `article_lead`, `caption_len_tokens`.

All NLP scripts use `en_core_web_trf` (transformer-backed). All scripts accept `--input`, `--output`, `--model`, `--batch-size`, and `--limit` (debug) arguments.

### Stage 2 — `src/structural_features.py`

Extracts features from **captions only**, except the four POS-overlap columns which compare caption lemmas against article lemmas:

- **Length**: `token_count`, `word_count`, `char_count`, `sent_count`
- **Complexity**: `avg_sent_len`, `dep_depth` (max tree depth), `clause_count` (clausal dep relations + 1)
- **Vocabulary**: `ttr` (type-token ratio on alpha tokens), `content_word_prop` (NOUN/VERB/ADJ/ADV proportion), `propn_prop`
- **POS overlap**: `noun_overlap`, `verb_overlap`, `adj_overlap`, `propn_overlap` — |cap lemmas ∩ art lemmas| / |cap lemmas| per POS tag; 0.0 if caption set is empty

### Stage 3 — `src/similarity.py`

Adds `tfidf_sim`, `jaccard_sim`, `sbert_sim`. spaCy runs once to lemmatise and remove stopwords/punctuation; the resulting lemma texts and lemma sets are shared by TF-IDF and Jaccard. SBERT receives raw text.

- `tfidf_sim`: TfidfVectorizer (bigrams, min_df=2, max_df=0.95) fit on combined lemmatised caption+article corpus; per-row cosine similarity.
- `jaccard_sim`: |C∩A|/|C∪A| on lemma sets from the same spaCy pass; 0.0 if union is empty.
- `sbert_sim`: `all-MiniLM-L6-v2` embeddings; articles truncated to **128 tokens** before encoding (TF-IDF and Jaccard use the full 512-token `article_lead`).

Accepts `--caption-col` (default `caption`) and `--article-col` (default `article_lead`) to handle CSVs with different column names. Prints per-method descriptive stats and mean by source/category after writing output.

### Stage 4 — `src/entity_alignment.py`

NER via `en_core_web_trf`; entity types: PERSON, GPE, ORG, DATE. Fuzzy entity matching uses Levenshtein similarity ≥ 0.85 (rapidfuzz). Adds `person_overlap`, `gpe_overlap`, `org_overlap`, `date_overlap` (per-type |E_cap ∩ E_art| / |E_cap|), `entity_jaccard`, and `entity_coverage` (overall metrics across all types).

### Stage 5 — `src/caption_classifier.py`

Placeholder — adds `caption_type = "unclassified"` for every row. The `classify()` function must be implemented once category definitions are finalised.

### `run_pipeline.py`

Runs all four stages sequentially. Intermediate CSVs are written alongside the final output (same directory, `_step1_`…`_step3_` suffixes) for inspection. Prints a summary of row count, mean/min/max of each similarity score, and `caption_type` distribution.

### Data layout

- `data/samples/sample_data_clean.csv` — cleaned 5k-row sample (no empty article_lead), committed and ready to use.
- `data/processed/` and `results/` — gitignored output directories.