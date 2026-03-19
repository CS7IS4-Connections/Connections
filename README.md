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
