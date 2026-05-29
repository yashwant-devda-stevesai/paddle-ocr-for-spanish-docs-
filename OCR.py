import os
import json
import time
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "5"   # adjust based on CPU cores
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

from paddleocr import PaddleOCRVL

INPUT_PATH = r"document_folder"   # image or pdf path
OUTPUT_DIR = "raw_result"

MODEL_VERSION = "v1.5"
DEVICE = "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)

input_file = Path(INPUT_PATH)

if not input_file.exists():
    raise FileNotFoundError(f"File not found: {INPUT_PATH}")

SUPPORTED = {
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".pdf"
}

if input_file.suffix.lower() not in SUPPORTED:
    raise ValueError(
        f"Unsupported file type: {input_file.suffix}\n"
        f"Supported: {', '.join(sorted(SUPPORTED))}"
    )

# LOAD MODEL
print("Loading PaddleOCR-VL model...")
start_load = time.time()

pipeline = PaddleOCRVL(
    pipeline_version=MODEL_VERSION,
    device=DEVICE
)

print(f"Model loaded in {time.time() - start_load:.2f} sec")
print(f"Processing file: {INPUT_PATH}")

start_pred = time.time()
results = pipeline.predict(INPUT_PATH)
print(f"OCR completed in {time.time() - start_pred:.2f} sec")

# TEXT EXTRACTOR (schema-aware)
def normalize_result_json(result_json):
    """
    PaddleOCR's in-memory res.json is usually {"res": {...}}, while the
    saved JSON file contains only the inner {...}. Accept both shapes.
    """
    if not isinstance(result_json, dict):
        return {}
    if "parsing_res_list" in result_json:
        return result_json
    if isinstance(result_json.get("res"), dict):
        return result_json["res"]
    return {}


def extract_text(result_json):
    """
    Build a clean raw OCR txt dump from parsing blocks only.
    Avoids recursive schema noise (labels list, debug strings, metadata).
    """
    lines = []
    result_json = normalize_result_json(result_json)
    blocks = result_json.get("parsing_res_list", [])
    for block in blocks:
        label = str(block.get("block_label", "")).strip()
        content = str(block.get("block_content", "")).strip()
        if not content:
            continue
        if label:
            lines.append(label)
        lines.extend(content.splitlines())
    return lines

for i, res in enumerate(results, start=1):
    print(f"\n========== PAGE / RESULT {i} ==========")
    res.print()
    # Save JSON
    json_path = os.path.join(OUTPUT_DIR, f"result_{i}.json")
    result_json = normalize_result_json(res.json)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result_json, f, ensure_ascii=False, indent=4)
    # Save TXT
    lines = extract_text(result_json)

    txt_path = os.path.join(OUTPUT_DIR, f"result_{i}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Saved JSON: {json_path}")
    print(f"Saved TXT : {txt_path} ({len(lines)} lines)")

print("\nDone!")

print("Testing for git")