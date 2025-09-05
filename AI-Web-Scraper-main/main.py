# main.py
import re, json
import streamlit as st
import pandas as pd
from io import StringIO

from scrape import (
    scrape_website,
    extract_body_content,
    clean_body_content,
    split_dom_content,
)
from parse import parse_with_ollama

# ----------------- Cleaning / Normalization helpers -----------------

PREFERRED_COLS = ["name", "price", "image_url", "url", "category", "brand", "sku"]

def _strip_code_fences(t: str) -> str:
    t = t.strip()
    return re.sub(r"^```.*?\n|\n```$", "", t, flags=re.DOTALL).strip() if t.startswith("```") else t

def _parse_json_first(text: str) -> pd.DataFrame | None:
    t = _strip_code_fences(text)
    # 1) Whole payload as JSON
    try:
        obj = json.loads(t)
        if isinstance(obj, list):
            return pd.DataFrame(obj)
        if isinstance(obj, dict):
            return pd.json_normalize(obj)
    except Exception:
        pass
    # 2) Merge many small JSON objects found in text
    objs = []
    for m in re.finditer(r"\{[^{}]*\}", t):
        try:
            objs.append(json.loads(m.group(0)))
        except Exception:
            pass
    if objs:
        return pd.DataFrame(objs)
    # 3) Arrays embedded in text
    arrays = re.findall(r"\[[\s\S]*?\]", t)
    rows = []
    for arr in arrays:
        try:
            part = json.loads(arr)
            if isinstance(part, list):
                rows.extend(part)
        except Exception:
            continue
    if rows:
        return pd.DataFrame(rows)
    return None

def _parse_csv_or_markdown(text: str) -> pd.DataFrame | None:
    t = _strip_code_fences(text)
    # CSV
    try:
        df = pd.read_csv(StringIO(t))
        if df.shape[1] > 1:
            return df
    except Exception:
        pass
    # Markdown table → DataFrame
    if "|" in t:
        lines = [ln for ln in t.splitlines() if ln.strip()]
        rows = [ln.strip() for ln in lines if "|" in ln]
        align_re = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$")
        cleaned = []
        for ln in rows:
            if align_re.match(ln):
                continue
            if ln.startswith("|"): ln = ln[1:]
            if ln.endswith("|"): ln = ln[:-1]
            cleaned.append(ln)
        if cleaned:
            md_like_csv = "\n".join(cleaned)
            try:
                df = pd.read_csv(StringIO(md_like_csv), sep="|")
                if df.shape[1] > 1:
                    return df
            except Exception:
                pass
    return None

def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip().lower() for c in df.columns]
    rename_map = {
        "product": "name", "title": "name",
        "link": "url", "product_url": "url",
        "image": "image_url", "img": "image_url", "imageurl": "image_url", "picture": "image_url",
        "cost": "price", "price_text": "price", "amount": "price",
        "cat": "category", "type": "category",
    }
    df = df.rename(columns={c: rename_map.get(c, c) for c in df.columns})
    for c in PREFERRED_COLS:
        if c not in df.columns:
            df[c] = ""
    other = [c for c in df.columns if c not in PREFERRED_COLS]
    return df[PREFERRED_COLS + other]

def _drop_header_rows_and_notes(df: pd.DataFrame) -> pd.DataFrame:
    header_set = set(df.columns)
    mask_header_dup = df.apply(lambda r: set(str(v).strip().lower() for v in r.values) == header_set, axis=1)
    df = df.loc[~mask_header_dup].copy()

    note_re = re.compile(r"\b(dummy|example|sample|n/?a|not available|missing|no price|no data)\b", re.I)
    def is_note_row(row):
        text = " ".join(str(v) for v in row.values)
        return bool(note_re.search(text))
    df = df[~df.apply(is_note_row, axis=1)].copy()
    return df

def _clean_price_column(df: pd.DataFrame) -> pd.DataFrame:
    def to_number(val):
        if pd.isna(val): return ""
        s = str(val)
        s = re.sub(r"[^\d\.,\-]", "", s)  # keep digits, dot, comma, minus
        s = s.replace(",", "")            # treat comma as thousands sep
        try:
            return float(s) if s else ""
        except Exception:
            return ""
    if "price" in df.columns:
        df["price"] = df["price"].apply(to_number)
    return df

def text_to_clean_df(text: str) -> pd.DataFrame:
    df = _parse_json_first(text)
    if df is None:
        df = _parse_csv_or_markdown(text)
    if df is None:
        lines = [ln for ln in (text or "").splitlines() if ln.strip()]
        df = pd.DataFrame({"value": lines})  # last resort

    df = _standardize_columns(df)
    df = _drop_header_rows_and_notes(df)
    df = _clean_price_column(df)
    df = df.dropna(how="all", subset=PREFERRED_COLS)  # drop empty rows across key cols
    df = df.drop_duplicates()
    return df.reset_index(drop=True)

# ----------------- Streamlit UI -----------------

st.title("AI Web Scraper")

url = st.text_input("Enter Website URL")

# Step 1 — Scrape
if st.button("Scrape Website") and url:
    st.write("Scraping the website...")
    dom_content = scrape_website(url)
    body_content = extract_body_content(dom_content)
    cleaned_content = clean_body_content(body_content)
    st.session_state.dom_content = cleaned_content

    with st.expander("View DOM Content"):
        st.text_area("DOM Content", cleaned_content, height=300)

# Step 2 — Parse with AI → Clean table → Download CSV
if "dom_content" in st.session_state:
    parse_description = st.text_area("Describe what you want to parse")

    if st.button("Parse Content") and parse_description:
        st.write("Parsing the content...")
        dom_chunks = split_dom_content(st.session_state.dom_content)

        # Ask the model (JSON-first in parse.py), then clean/normalize here
        result_text = parse_with_ollama(dom_chunks, parse_description)
        st.session_state.last_result_text = result_text

        df = text_to_clean_df(result_text)

        # Put preferred columns first (already done), show, and allow download
        st.dataframe(df, use_container_width=True)

        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download CSV",
            data=csv_bytes,
            file_name="scraped_data.csv",
            mime="text/csv",
        )
