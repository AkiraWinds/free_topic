'''
chunking by division
how to run: python3 chunking_division.py
'''
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Union


DIVISION_PATTERN = re.compile(
    r"(\[DIVISION\]\s+Division\s+\d+[^[]*)",
    flags=re.IGNORECASE
)


def chunk_by_division(text: str) -> List[Dict[str, str]]:
    """
    Split text by [DIVISION].

    Returns:
    - If divisions exist:
        [
          {"division_title": "...", "text": "..."},
          ...
        ]
    - If no division exists:
        [
          {"division_title": None, "text": original_text}
        ]
    """
    if not text:
        return []

    divisions = DIVISION_PATTERN.split(text)

    # No division found
    if len(divisions) == 1:
        return [
            {
                "division_title": None,
                "text": text.strip()
            }
        ]

    chunks: List[Dict[str, str]] = []

    # ["", "[DIVISION] Division 1 ...", "content", ...]
    for i in range(1, len(divisions), 2):
        header = divisions[i].strip()
        body = divisions[i + 1].strip() if i + 1 < len(divisions) else ""

        division_title = header.replace("[DIVISION]", "").strip()

        chunks.append(
            {
                "division_title": division_title,
                "text": body
            }
        )

    return chunks


def process_records_by_division(
    records: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Read records one by one and split each by Division.
    """
    output_records: List[Dict[str, Any]] = []

    for rec in records:
        original_text = rec.get("text", "")
        division_chunks = chunk_by_division(original_text)

        # Keep all original metadata except text
        base_meta = {
            k: v for k, v in rec.items() if k != "text"
        }

        for idx, chunk in enumerate(division_chunks):
            new_rec = dict(base_meta)

            new_rec["division_title"] = chunk["division_title"]
            new_rec["text"] = chunk["text"]

            # Build stable id
            base_id = rec.get("id", "unknown")
            new_rec["id"] = f"{base_id}_division_{idx + 1}"

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
):
    input_path = Path(input_path)
    output_path = Path(output_path)

    data = load_json(input_path)

    if not isinstance(data, list):
        raise ValueError("Expected input JSON to be a list of records")

    processed = process_records_by_division(data)
    save_json(processed, output_path)


# =========================
# Example usage
# =========================
if __name__ == "__main__":
    main(
        input_path="../data/processed/act-2011-070/act-2011-070_chunked_part.json",
        output_path="../data/processed/act-2011-070/act-2011-070_chunked_division.json"
    )
