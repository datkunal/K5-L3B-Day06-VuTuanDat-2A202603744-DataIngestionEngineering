"""Download the corpus PDF (CC BY 4.0) to data/raw/ and verify its SHA-256 against the manifest.

Run from starter/:  python scripts/download_corpus.py
"""
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "starter"))
from src import config  # noqa: E402


def main() -> None:
    config.RAW_PDF.parent.mkdir(parents=True, exist_ok=True)
    if not config.RAW_PDF.exists():
        print(f"Downloading {config.SOURCE_URL}")
        urllib.request.urlretrieve(config.SOURCE_URL, config.RAW_PDF)
    digest = hashlib.sha256(config.RAW_PDF.read_bytes()).hexdigest()
    print("sha256", digest)
    if config.MANIFEST.exists():
        expected = json.loads(config.MANIFEST.read_text(encoding="utf-8"))["sources"][0]["version_or_snapshot"]
        if digest not in expected:
            print("WARNING: PDF differs from the version recorded in data/manifest.json; "
                  "page numbers and benchmark evidence may no longer match.")
        else:
            print("matches manifest")


if __name__ == "__main__":
    main()
