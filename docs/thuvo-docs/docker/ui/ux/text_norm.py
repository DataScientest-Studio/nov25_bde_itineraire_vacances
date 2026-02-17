# ======================================================================================================================
# FILTER CONTEXT ---> docker/ui/ux/text_norm.py
# ======================================================================================================================

import re
import unicodedata
from infra.imports import pd

def to_str_series(s: pd.Series) -> pd.Series:
    return s.astype("string").fillna("").str.strip()

def norm_txt(x: str) -> str:
    x = (x or "").strip().lower()
    x = "".join(c for c in unicodedata.normalize("NFD", x) if unicodedata.category(c) != "Mn")
    x = x.replace("-", " ").replace("’", "'")
    x = re.sub(r"\s+", " ", x)
    return x
