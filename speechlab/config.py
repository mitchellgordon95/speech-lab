import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
DATA = Path(os.environ.get("SPEECHLAB_DATA", ROOT / "data"))
CACHE = ROOT / ".cache"
for path in (DATA, DATA / "clips", DATA / "features", DATA / "results", DATA / "axes", CACHE):
    path.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(CACHE / "huggingface"))
os.environ.setdefault("MPLCONFIGDIR", str(CACHE / "matplotlib"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
MAX_SECONDS = 30
MAX_BYTES = 25 * 1024 * 1024
