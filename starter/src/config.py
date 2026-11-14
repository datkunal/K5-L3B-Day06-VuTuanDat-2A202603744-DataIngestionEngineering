"""Single place for corpus, model, store and strategy settings (kept identical across A/B/C)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # repository root
DATA_DIR = ROOT / "data"
RAW_PDF = DATA_DIR / "raw" / "introduction-intellectual-property.pdf"
PAGES_JSONL = DATA_DIR / "processed" / "pages.jsonl"
MANIFEST = DATA_DIR / "manifest.json"
BENCHMARK = ROOT / "benchmark" / "questions.json"
RESULTS = ROOT / "results"
def _chroma_dir() -> Path:
    """Persistent Chroma directory.

    Chroma 1.5.9 on Windows cannot RE-OPEN an HNSW index stored under a non-ASCII path
    (observed: 'Error loading hnsw index' when the repo lives in 'D:/AI thực chiến/...').
    So: $DAY06_CHROMA_DIR if set; else <repo>/storage/chroma if that path is ASCII;
    else %LOCALAPPDATA%/day06_chroma (or ~/.day06_chroma).
    """
    env = os.getenv("DAY06_CHROMA_DIR")
    if env:
        return Path(env)
    default = ROOT / "storage" / "chroma"
    if str(default).isascii():
        return default
    base = os.getenv("LOCALAPPDATA")
    return Path(base) / "day06_chroma" if base else Path.home() / ".day06_chroma"


CHROMA_DIR = _chroma_dir()                            # persistent Chroma database (not in git)

DOCUMENT_ID = "openstax-intro-ip-2021"
SOURCE_URL = ("https://assets.openstax.org/oscms-prodcms/media/documents/"
              "introduction-intellectual-property_-_WEB.pdf")
SOURCE = "data/raw/introduction-intellectual-property.pdf"

# Physical (1-based) PDF pages counted as content: chapter bodies only. Excluded: cover,
# front matter/preface (1-16), chapter-opener figure-only pages (17, 65, 101, 171),
# assessment questions (60-64, 97-100, 135-140, 165-170, 188-190), glossary, answer key, index.
CHAPTER_PAGE_RANGES = {1: (18, 59), 2: (66, 96), 3: (102, 134), 4: (141, 164), 5: (172, 187)}
CONTENT_PAGES = [p for a, b in CHAPTER_PAGE_RANGES.values() for p in range(a, b + 1)]

EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
EMBEDDING_REVISION = None  # resolved commit hash is recorded at run time
QUERY_PREFIX = "query: "      # e5 models require these prefixes
PASSAGE_PREFIX = "passage: "
BATCH_SIZE = 16

STRATEGIES = {
    "A": {"chunking": "fixed_size", "size": 500, "overlap": 50, "use_filter": False},
    "B": {"chunking": "paragraph", "max_size": 1000, "use_filter": False},
    "C": {"chunking": "paragraph", "max_size": 1000, "use_filter": True},  # same chunks as B
}
TOP_K = 3
