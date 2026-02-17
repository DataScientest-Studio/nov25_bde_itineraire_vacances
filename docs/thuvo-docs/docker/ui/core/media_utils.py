# ======================================================================================================================
# FILTER CONTEXT ---> docker/ui/core/media_utils.py
# ======================================================================================================================

import re
import pandas as pd

def is_valid_http_url(x) -> bool:
    if x is None:
        return False
    # gère pandas NA / NaN
    try:
        if pd.isna(x):
            return False
    except Exception:
        pass

    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "<na>", "none", "null"}:
        return False

    return bool(re.match(r"^https?://", s))


def is_image_url(x) -> bool:
    if not is_valid_http_url(x):
        return False
    s = str(x).strip().lower()
    return s.endswith((".jpg", ".jpeg", ".png", ".webp"))