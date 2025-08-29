#!/usr/bin/env python3
"""
Preprocess Flickr8k captions files and create tokenizer + sequences.
Outputs:
 - captions_dict.pkl: {image_id: [caption1, caption2, ...]}
 - tokenizer.json: Keras tokenizer JSON
 - sequences.pkl: dict with 'train_seqs' and metadata required for training

Usage:
    python src/preprocess_captions.py --captions_file data/Flickr8k_text/Flickr8k.token.txt --out_dir data/
"""
from __future__ import annotations
import argparse, re, json, pickle
from pathlib import Path
from collections import defaultdict

import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def load_captions(path):
    # Each line: <image_id>#<caption_idx>\t<caption text>
    captions = defaultdict(list)
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            img_cap, cap = line.split("\t")
            img_id = img_cap.split("#")[0]
            captions[img_id].append(cap.lower())
    return captions

def clean_caption(caption):
    # basic cleaning: remove non-alpha except spaces, collapse spaces
    caption = caption.replace("-", " ")
    caption = re.sub(r"[^a-z0-9\s']", " ", caption)
    caption = re.sub(r"\s+", " ", caption).strip()
    return caption

def build_tokenizer(captions_dict, vocab_size=10000):
    texts = []
    for caps in captions_dict.values():
        for c in caps:
            texts.append("startseq " + clean_caption(c) + " endseq")
    tokenizer = Tokenizer(num_words=vocab_size, oov_token="unk")
    tokenizer.fit_on_texts(texts)
    return tokenizer

def create_sequences(tokenizer, captions, max_len=None):
    seqs = []
    for cap in captions:
        seq = tokenizer.texts_to_sequences([cap])[0]
        for i in range(1, len(seq)):
            in_seq, out_seq = seq[:i], seq[i]
            seqs.append((in_seq, out_seq))
    if max_len is None:
        max_len = max(len(tokenizer.texts_to_sequences([c])[0]) for c in captions)
    return seqs, max_len

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--captions_file", type=str, required=True, help="Path to Flickr8k.token.txt")
    parser.add_argument("--out_dir", type=str, default="data", help="Output directory")
    parser.add_argument("--vocab_size", type=int, default=10000)
    args = parser.parse_args()

    captions = load_captions(args.captions_file)
    # convert keys to filenames used by the feature extractor (e.g., '1000268201_693b08cb0e.jpg')
    captions = {k if k.endswith(".jpg") else k + ".jpg": v for k, v in captions.items()}

    # clean and wrap sequences with startseq/endseq
    cap_wrapped = {img: ["startseq " + clean_caption(c) + " endseq" for c in caps] for img, caps in captions.items()}

    tokenizer = build_tokenizer(captions, vocab_size=args.vocab_size)
    tokenizer_json = tokenizer.to_json()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "captions_dict.pkl", "wb") as f:
        pickle.dump(cap_wrapped, f)
    with open(out_dir / "tokenizer.json", "w", encoding="utf8") as f:
        f.write(tokenizer_json)

    print(f"Saved captions_dict.pkl and tokenizer.json to {out_dir.resolve()}")
    print("Done.")

if __name__ == "__main__":
    main()
