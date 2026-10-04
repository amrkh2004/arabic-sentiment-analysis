"""
Large-Scale Arabic Sentiment Reviews Dataset Preparation Script (~25,000 - 30,000 Reviews)
Features:
- Pure Customer, E-Commerce, and Service Reviews (HARD + SHEIN + LABR)
- ~10,000 Positive, ~10,000 Negative, ~5,000+ Neutral
- Fully normalized and cleaned Arabic text
- Length-filtered for direct review signal
Saves directly to 'reviews.csv' for training and benchmarking.
"""

import os
import re
import sys
import pandas as pd
import numpy as np

def clean_arabic_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#\w+", " ", text)
    text = re.sub(r"[إأآا]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"ة", "ه", text)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    tashkeel = re.compile(r"[\u0617-\u061A\u064B-\u0652]")
    text = re.sub(tashkeel, "", text)
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"[^\u0600-\u06FF\s0-9.,!?،؟]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def is_valid_product_review(text: str) -> bool:
    if len(text) < 15 or len(text) > 350:
        return False
    bad_keywords = ["مقتل", "الجيش", "الحرب", "بشار", "الانقلاب", "وزير", "داعش", "النظام", "سياسة", "مفتي", "حلب"]
    for w in bad_keywords:
        if w in text:
            return False
    return True

def generate_large_dataset():
    print("=" * 65)
    print("Building Large-Scale Arabic Reviews Dataset (~25,000 - 30,000 Reviews)...")
    print("=" * 65)

    pos_list, neu_list, neg_list = [], [], []

    # 1. SHEIN E-Commerce Reviews
    try:
        print("[1/3] Loading E-Commerce Reviews (SHEIN)...")
        shein_url = "https://huggingface.co/datasets/Ruqiya/Arabic_Reviews_of_SHEIN/resolve/main/data/train-00000-of-00001.parquet"
        df_shein = pd.read_parquet(shein_url)
        for _, row in df_shein.iterrows():
            t = clean_arabic_text(str(row.get("text", "")))
            if is_valid_product_review(t):
                try:
                    r = float(row.get("label", 0))
                    if r <= 2: neg_list.append(t)
                    elif r == 3: neu_list.append(t)
                    elif r >= 4: pos_list.append(t)
                except:
                    pass
        print(f"  -> SHEIN samples: Pos={len(pos_list)}, Neu={len(neu_list)}, Neg={len(neg_list)}")
    except Exception as e:
        print("  -> SHEIN load error:", e)

    # 2. HARD (105k Customer & Booking Reviews)
    try:
        print("[2/3] Loading Customer Reviews from HARD Benchmark...")
        hard_url = "https://huggingface.co/datasets/Elnagara/hard/resolve/main/plain_text/train-00000-of-00001.parquet"
        df_hard = pd.read_parquet(hard_url)
        for _, row in df_hard.iterrows():
            t = clean_arabic_text(str(row.get("text", "")))
            if is_valid_product_review(t):
                lbl = row.get("label")
                if lbl in (0, 1) and len(neg_list) < 18000:
                    neg_list.append(t)
                elif lbl in (3, 4) and len(pos_list) < 18000:
                    pos_list.append(t)
        print(f"  -> Accumulated with HARD: Pos={len(pos_list)}, Neu={len(neu_list)}, Neg={len(neg_list)}")
    except Exception as e:
        print("  -> HARD load error:", e)

    # 3. LABR Curated 3-Star Neutral Reviews
    try:
        print("[3/3] Loading Customer Neutral Reviews...")
        labr_train_url = "https://huggingface.co/datasets/mohamedadaly/labr/resolve/main/plain_text/train-00000-of-00001.parquet"
        labr_test_url = "https://huggingface.co/datasets/mohamedadaly/labr/resolve/main/plain_text/test-00000-of-00001.parquet"
        df_labr = pd.concat([pd.read_parquet(labr_train_url), pd.read_parquet(labr_test_url)], ignore_index=True)
        for _, row in df_labr.iterrows():
            if row.get("label") == 2:  # 3-star neutral
                t = clean_arabic_text(str(row.get("text", "")))
                if 15 <= len(t) <= 250 and is_valid_product_review(t):
                    neu_list.append(t)
        print(f"  -> Total Neutral accumulated: {len(neu_list)}")
    except Exception as e:
        print("  -> Neutral load error:", e)

    # Deduplicate
    pos_unique = list(dict.fromkeys(pos_list))
    neu_unique = list(dict.fromkeys(neu_list))
    neg_unique = list(dict.fromkeys(neg_list))

    print(f"\nUnique pools: Pos={len(pos_unique):,}, Neu={len(neu_unique):,}, Neg={len(neg_unique):,}")

    target_pos = min(11000, len(pos_unique))
    target_neg = min(11000, len(neg_unique))
    target_neu = len(neu_unique)

    np.random.seed(42)
    selected_pos = np.random.choice(pos_unique, target_pos, replace=False)
    selected_neg = np.random.choice(neg_unique, target_neg, replace=False)
    selected_neu = np.array(neu_unique)

    df_pos = pd.DataFrame({"text": selected_pos, "label": "positive"})
    df_neg = pd.DataFrame({"text": selected_neg, "label": "negative"})
    df_neu = pd.DataFrame({"text": selected_neu, "label": "neutral"})

    final_df = pd.concat([df_pos, df_neg, df_neu], ignore_index=True)
    final_df = final_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    print("\n" + "=" * 65)
    print(f"Large Dataset Ready: {len(final_df):,} total reviews")
    print("Class Distribution:")
    print(final_df["label"].value_counts())
    print("=" * 65)

    out_file = "reviews.csv"
    final_df.to_csv(out_file, index=False, encoding="utf-8-sig")
    print(f"\nSaved successfully to: {os.path.abspath(out_file)}")
    return final_df

if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    generate_large_dataset()
