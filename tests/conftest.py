import sys
from pathlib import Path
from api.main import app

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
