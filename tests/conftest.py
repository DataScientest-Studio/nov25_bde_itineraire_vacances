import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # racine du repo
API_SRC = ROOT / "docs" / "thuvo-docs" / "src"

sys.path.insert(0, str(SRC))
