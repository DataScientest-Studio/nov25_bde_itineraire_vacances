import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
API_SRC = REPO_ROOT / "docs" / "thuvo-docs" / "src"

sys.path.insert(0, str(API_SRC))

assert API_SRC.exists(), f"API source path not found: {API_SRC}"
