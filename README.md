# Day 06 · Data Ingestion Engineering — bài làm

Kho tri thức truy xuất được trên sách **Introduction to Intellectual Property** (OpenStax, 2021, CC BY 4.0) và so sánh 3 chiến lược chuẩn bị dữ liệu A/B/C trên cùng corpus, model và bộ câu hỏi. Đề bài gốc: [ASSIGNMENT.md](ASSIGNMENT.md), [GUIDE.md](GUIDE.md), [BENCHMARK.md](BENCHMARK.md).

| Thành phần | Lựa chọn |
|---|---|
| Corpus | 1 PDF, 201 trang vật lý → **146 trang nội dung** (chương 1–5), 427.691 ký tự sau chuẩn hóa — [data/manifest.json](data/manifest.json) |
| Đọc PDF | `pypdfium2` 4.30.0 (pypdf làm vỡ khoảng trắng giữa chữ nên chỉ dùng để đọc bookmark) |
| Embedding | `intfloat/multilingual-e5-base` (Sentence Transformers 6.2.0, local, CPU), 768 chiều, tiền tố `query: ` / `passage: `, vector chuẩn hóa L2. Chi phí 0 USD |
| Vector store | **Chroma 1.5.9** `PersistentClient` (lưu trên đĩa), HNSW approximate, metric **cosine**, mỗi chiến lược một collection |
| Metadata mỗi chunk | `document_id`, `source`, `page` (trang PDF vật lý), `printed_page`, `chapter`, **`ip_type`** (patent/copyright/trademark/trade_secret — trường lọc), `section`, `section_title`, `char_start/end`, `content_sha1` |

Docker không có trên máy làm bài nên **không dùng pgvector**; các file `docker-compose.yml`, `init.sql`, `scripts/check_setup.py` của starter được giữ nguyên nhưng không dùng.

## Chạy lại

Yêu cầu Python 3.11–3.12, khoảng 1,5 GB cho torch + model (tải lần đầu), không cần API key.

```bash
cd starter
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1   |   macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

python scripts/download_corpus.py      # tải PDF vào data/raw/ và so SHA-256 với manifest
python -m src.prepare                  # đọc + chuẩn hóa -> data/processed/pages.jsonl, manifest, thống kê chunk
python -m src.verify_benchmark         # kiểm chứng evidence/filter của 24 câu TRƯỚC khi truy xuất
python -m src.run ingest               # embed + upsert A/B/C vào Chroma (chạy lại không nhân đôi)
python -m src.run query --split dev    # trace + Hit@1/Hit@3 trên dev
python -m src.run query --split test   # chỉ chạy sau khi đã chốt chiến lược bằng dev
python -m src.sample_coherence         # sinh mẫu 10 vị trí cho audit coherence
```

**Lưu ý Windows:** Chroma 1.5.9 không mở lại được index HNSW nằm dưới đường dẫn có ký tự Unicode (repo này nằm trong `D:/AI thực chiến/…`). Vì vậy index được lưu ở `$DAY06_CHROMA_DIR` nếu đặt biến này; nếu không thì ở `storage/chroma` khi đường dẫn repo là ASCII, còn lại ở `%LOCALAPPDATA%/day06_chroma`. Chi tiết trong [results/ingestion_errors.json](results/ingestion_errors.json).

## Cấu trúc

```text
data/manifest.json              nguồn, giấy phép, SHA-256, quy tắc đếm trang, số ký tự từng trang
data/processed/pages.jsonl      146 trang đã chuẩn hóa + metadata (PDF gốc không commit: data/raw/ bị ignore)
benchmark/questions.json        24 câu (18 dev / 6 test), gold page, evidence, filter
starter/src/                    ingestion, chunking, embeddings, store (Chroma), evaluation, run, ...
artifacts/experiment_log.csv    cấu hình + giả thuyết + kết quả A/B/C
results/                        manifest kiểm chứng benchmark, ingest log, trace top-3, metrics, nhãn thủ công
report/individual/              báo cáo cá nhân
```

## Kết quả chính

Chiến lược được chọn trên dev trước khi chạy test là **C** (chunk theo đoạn + filter `ip_type`), theo [results/strategy_choice.json](results/strategy_choice.json).

| Strategy | Chunks | Dev Hit@1 | Dev Evidence Hit@3 | Test Hit@1 | Test Evidence Hit@3 | Coherence /2 | Grounding test /2 |
|---|---|---|---|---|---|---|---|
| A fixed 500/50 | 1008 | 10/18 | 16/18 | 3/6 | 5/6 | 0,9 | 2,00 |
| B paragraph ≤1000 | 599 | 13/18 | 17/18 | 4/6 | 5/6 | 1,8 | 1,83 |
| C = B + filter | 599 | 14/18 | 17/18 | 5/6 | 5/6 | 1,8 | 2,00 |

Nạp lại lần hai: 0 chunk phải embed lại, số bản ghi không đổi (1008/599/599), không có bản ghi trùng ([results/ingest_log.jsonl](results/ingest_log.jsonl)). Khi tài liệu thay đổi, chunk có `content_sha1` khác sẽ được embed và upsert lại, còn chunk ID không còn tồn tại thì bị xóa (`src/run.py::ingest`). Phân tích đầy đủ trong [báo cáo cá nhân](report/individual/2A202603744_VuTuanDat.md).
