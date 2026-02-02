import json
from pathlib import Path
from typing import Any, Dict, List, Union

import tiktoken


# =========================
# Config
# =========================

ENCODING_NAME = "cl100k_base"
TEXT_KEY = "text"
ID_KEY = "id"

encoding = tiktoken.get_encoding(ENCODING_NAME)


# =========================
# Utils
# =========================

def count_tokens(text: str) -> int:
    if not isinstance(text, str) or not text.strip():
        return 0
    return len(encoding.encode(text))


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_records(data: Any) -> List[Dict[str, Any]]:
    """
    Normalize JSON layouts into a list of record dicts.
    Expected chunked format: List[Dict]
    """
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]

    if isinstance(data, dict):
        if isinstance(data.get("data"), list):
            return [x for x in data["data"] if isinstance(x, dict)]
        return [data]

    return []


# =========================
# Core logic
# =========================

def count_tokens_per_record(
    json_path: Path,
    text_key: str = TEXT_KEY,
    id_key: str = ID_KEY,
) -> List[Dict[str, Any]]:
    """
    Count tokens per chunk (record) in one JSON file.
    """
    data = load_json(json_path)
    records = extract_records(data)

    results: List[Dict[str, Any]] = []

    for idx, rec in enumerate(records):
        text = rec.get(text_key, "")
        tokens = count_tokens(text)

        record_id = rec.get(id_key, f"record_{idx}")

        results.append({
            "file": json_path.name,
            "record_id": record_id,
            "tokens": tokens,
        })

    return results


def process_folder_per_record(
    input_folder: Union[str, Path],
    text_key: str = TEXT_KEY,
    id_key: str = ID_KEY,
    exclude_name_contains: List[str] | None = None,
) -> List[Dict[str, Any]]:
    """
    Recursively process all JSON files and count tokens per record.
    """
    input_folder = Path(input_folder)
    exclude_name_contains = exclude_name_contains or []

    all_results: List[Dict[str, Any]] = []

    json_files = sorted(input_folder.rglob("*.json"))

    print(f"[INFO] Found {len(json_files)} JSON files (recursive)")

    for json_file in json_files:
        if any(substr in json_file.name for substr in exclude_name_contains):
            continue

        try:
            per_file_results = count_tokens_per_record(
                json_file,
                text_key=text_key,
                id_key=id_key,
            )
            all_results.extend(per_file_results)

        except Exception as e:
            print(f"⚠️ Failed to process {json_file}: {e}")

    return all_results


# =========================
# Output
# =========================

def write_json_report(results: List[Dict[str, Any]], output_path: Union[str, Path]) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def write_txt_report(results: List[Dict[str, Any]], output_path: Union[str, Path]) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(
                f"{r['file']}\t{r['record_id']}\t{r['tokens']} tokens\n"
            )


# =========================
# Main
# =========================

if __name__ == "__main__":
    INPUT_FOLDER = "../data/processed/"
    OUTPUT_JSON = "../data/reports/token_per_chunk.json"
    OUTPUT_TXT  = "../data/reports/token_per_chunk.txt"

    results = process_folder_per_record(
        INPUT_FOLDER,
        text_key=TEXT_KEY,
        id_key=ID_KEY,
        exclude_name_contains=["token_report", "report"],
    )

    write_json_report(results, OUTPUT_JSON)
    write_txt_report(results, OUTPUT_TXT)

    print("✅ Per-chunk token counting completed.")
    print(f"📊 Total chunks: {len(results)}")
    print(f"📄 JSON report: {OUTPUT_JSON}")
    print(f"📄 TXT report:  {OUTPUT_TXT}")
