"""Import the upstream model without editing or installing v1."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "v1" / "llada"))
