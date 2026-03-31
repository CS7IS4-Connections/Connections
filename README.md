# Connections

Text analytics project analysing caption-article textual similarity using the
[VisualNews dataset](https://huggingface.co/datasets/twelcone/VisualNews).

## Install

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run preprocessing

```bash
# Sample (5 000 rows) — fast, for development
python src/preprocessing.py --mode sample --n 5000 --output data/samples/sample_5k.csv

# Full dataset
python src/preprocessing.py --mode full --output data/processed/full.csv
```

The script:
1. Loads VisualNews from Hugging Face (no manual download needed)
2. Prints the dataset schema and unique category values for transparency
3. Cleans text (unicode, HTML, whitespace)
4. Truncates articles to the first 512 whitespace tokens
5. Writes a CSV with columns:
   `item_id, source, category, caption, article_lead, caption_len_tokens`

## Structural features (Task 2)

Requires a preprocessing CSV from the step above. Install the English pipeline once:

```bash
python -m spacy download en_core_web_sm
```

```bash
python src/structural_features.py \
    --input data/samples/sample_5k.csv \
    --output results/csv/sample_5k_structural.csv
```

This adds columns for each of `--text-cols` (default `caption,article_lead`):

- `{col}_char_len`, `{col}_token_len` — character count and spaCy token count (non-space tokens)
- `{col}_ttr` — type–token ratio on alphabetic tokens (lowercased surface form); NaN if none
- `{col}_dep_depth_max` — maximum dependency depth to root, max over sentences
- `{col}_pos_ADJ` … `{col}_pos_X` — proportion of tokens (non-space) with each Universal POS tag; NaN if the text has no tokens

Empty or missing article text yields zeros/NaNs consistent with the above. Use `item_id` to merge with other feature tables.
