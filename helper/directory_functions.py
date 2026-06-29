import os
import re

import pandas as pd


def is_dataset_dir_existing(dataset_dir_name: str) -> bool:
    """
    checks if dataset directory (dataset_dir_name) exists in "./data/dataset"
    """
    root = get_root()

    # get complete path to dataset
    dataset_path = os.path.join("data", "dataset")
    searched_dir = os.path.join(root, dataset_path)
    if os.listdir(searched_dir).count(dataset_dir_name) == 1:
        return True
    else:
        return False


def get_root() -> str:
    """get project root"""
    project_name = "EMI_Project"
    cwd = os.getcwd()
    cwd_list = cwd.split(project_name)
    root = os.path.join(cwd_list[0], project_name)

    return root


def create_dir_name(dataset_name):
    """
    Takes name of dataset, turns it into snake_case directory name

    e.g.:
    "Leonardo6/memotion" -> Leonardo6_memotion_
    """
    name_list = dataset_name.split("/")
    dir_name = ""
    for items in name_list:
        dir_name += (items + "_")

    return dir_name


def search_memotion_dataset_7k_dir():
    target = "memotion_dataset_7k"
    ret_val = search_dir(target)
    if ret_val is not None:
        return ret_val
    return None


def search_dir(searched_dir):
    root = get_root()

    for current_dir, dirs, files in os.walk(root):
        if searched_dir in dirs:
            path = os.path.join(current_dir, searched_dir)
            return path

    return None


def is_memotion_dataset_7k_existing() -> bool:
    if search_memotion_dataset_7k_dir() is not None:
        return True
    else:
        return False


def data_cleaning_and_label_encoding(self, df: pd.DataFrame, img_dir):
    # ── Fix image names ──
    df["image_name"] = df["image_name"].astype(str).str.strip()

    def fix_ext(name):
        if not name.lower().endswith((".jpg", ".jpeg", ".png", ".gif")):
            return name + ".jpg"
        return name

    df["image_name"] = df["image_name"].apply(fix_ext)

    # ── Drop rows with missing images ──
    df["img_ok"] = df["image_name"].apply(
        lambda x: os.path.exists(os.path.join(img_dir, x))
    )
    print(f"Images found: {df['img_ok'].sum()} / {len(df)}")
    df = df[df["img_ok"]].reset_index(drop=True)

    # ── Text column ──
    text_col = "text_corrected" if "text_corrected" in df.columns else "text_ocr"
    print(f"Text column: {text_col}")

    def clean_text(t):
        if pd.isna(t) or str(t).strip() in ("", "nan"):
            return "no text"
        t = str(t).lower()
        t = re.sub(r"http\S+", "", t)
        t = re.sub(r"[^a-zA-Z\s!?]", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
        return t or "no text"

    df["clean_text"] = df[text_col].apply(clean_text)

    # ── Label maps ──
    SENTIMENT_MAP = {
        "very_positive": "positive", "positive": "positive",
        "neutral": "neutral",
        "negative": "negative", "very_negative": "negative"
    }
    HUMOUR_MAP = {"not_funny": 0, "funny": 1, "very_funny": 2, "hilarious": 3}
    SARCASM_MAP = {"not_sarcastic": 0, "general": 1, "twisted_meaning": 2, "very_twisted": 3}
    OFFENSIVE_MAP = {"not_offensive": 0, "slight": 1, "very_offensive": 2, "hateful_offensive": 3}
    SENTIMENT_MAP3 = {"positive": 0, "neutral": 1, "negative": 2}

    df["sentiment_3"] = df["overall_sentiment"].str.strip().str.lower().map(SENTIMENT_MAP)
    df = df.dropna(subset=["sentiment_3"]).reset_index(drop=True)
    df["label_sentiment"] = df["sentiment_3"].map(SENTIMENT_MAP3)

    def safe_map(col, mapping, default=0):
        if col not in df.columns:
            return pd.Series([default] * len(df))
        return df[col].str.strip().str.lower().map(mapping).fillna(default).astype(int)

    df["label_humour"] = safe_map("humour", HUMOUR_MAP)
    df["label_sarcasm"] = safe_map("sarcasm", SARCASM_MAP)
    df["label_offensive"] = safe_map("offensive", OFFENSIVE_MAP)

    NUM_CLASSES = {
        "sentiment": 3,
        "humour": 4,
        "sarcasm": 4,
        "offensive": 4,
    }

    print(f"Final dataset: {len(df)} rows")
    print(df["sentiment_3"].value_counts())
    return df
