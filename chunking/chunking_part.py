'''
chunking for legislative documents
based on PART headers
how to run: python3 chunking.py
'''

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Union
import uuid


# =========================
# Pattern for PART headers
# =========================
PART_PATTERN = re.compile(
    r"(\[PART\]\s+Part\s+\d+[^[]*)",
    flags=re.IGNORECASE
)


def chunk_by_part(cleaned_text: str) -> List[Dict[str, str]]:
    """
    Split cleaned text into PART-level chunks.

    Returns:
        [
            {
                "part_title": "...",
                "text": "..."
            },
            ...
        ]
    """
    if not cleaned_text:
        return []

    parts = PART_PATTERN.split(cleaned_text)
    chunks: List[Dict[str, str]] = []

    for i in range(1, len(parts), 2):
        part_header = parts[i].strip()
        part_body = parts[i + 1].strip() if i + 1 < len(parts) else ""

        part_title = part_header.replace("[PART]", "").strip()

        chunks.append(
            {
                "part_title": part_title,
                "text": part_body
            }
        )

    return chunks


def process_records(
    records: List[Dict[str, Any]],
    normalized_text_key: str = "text_normalized"
) -> List[Dict[str, Any]]:
    """
    For each original record:
    - chunk by PART
    - emit one new record per PART
    - keep ALL original keys EXCEPT 'text' and 'text_normalized'
    - add: id, part_title, text
    """
    output_records: List[Dict[str, Any]] = []

    for rec in records:
        cleaned_text = rec.get(normalized_text_key, "")
        part_chunks = chunk_by_part(cleaned_text)

        # Keep all original metadata except text fields
        base_meta = {
            k: v
            for k, v in rec.items()
            if k not in {"text", "text_normalized"}
        }

        for idx, chunk in enumerate(part_chunks):
            new_rec = dict(base_meta)

            # New unique chunk id
            new_rec["id"] = f"{rec.get('version_id', 'unknown')}_part_{idx+1}"

            new_rec["part_title"] = chunk["part_title"]
            new_rec["text"] = chunk["text"]

            output_records.append(new_rec)

    return output_records


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main(
    input_path: Union[str, Path],
    output_path: Union[str, Path]
) -> None:
    input_path = Path(input_path)
    output_path = Path(output_path)

    data = load_json(input_path)

    # Root is a list
    if isinstance(data, list):
        processed = process_records(data)
        save_json(processed, output_path)
        return

    # Root is a dict with "data"
    if isinstance(data, dict) and isinstance(data.get("data"), list):
        processed = process_records(data["data"])
        save_json(processed, output_path)
        return

    raise ValueError(
        "Unsupported JSON structure. Expected a list or a dict with a 'data' list."
    )


# =========================
# Example usage
# =========================
if __name__ == "__main__":
    main(
        input_path="../data/processed/act-2011-070/act-2011-070_normalized.json",
        output_path="../data/processed/act-2011-070/act-2011-070_chunked.json"
    )
