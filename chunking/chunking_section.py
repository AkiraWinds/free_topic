import json
import re
from pathlib import Path
from typing import Any, Dict, List, Union


# 1) 用于“切块”：抓到每个 [SECTION] 的起始位置
#    我们用 lookahead 保留分隔点，便于 split
SECTION_SPLIT_PATTERN = re.compile(r"(?=\[SECTION\]\s+)", flags=re.IGNORECASE)

# 2) 用于“解析 section header”：从块开头提取编号和剩余文本
#    示例: "[SECTION] 21 Excluded provisions (1) For the purposes..."
#    -> group(1)= "21" (或 "9A", "29CA"), group(2)= "Excluded provisions (1) For the purposes..."
SECTION_HEADER_PATTERN = re.compile(
    r"^\[SECTION\]\s+(\d+[A-Z]{0,3})\s*(.*)$",
    flags=re.IGNORECASE
)


def chunk_by_section(text: str) -> List[Dict[str, str]]:
    """
    Split text into SECTION-level chunks.

    For each chunk:
      - section_title: the numeric/alpha section id right after [SECTION] (e.g., 21, 9A, 29CA)
      - text: everything after that id (plus any following content until next [SECTION])

    If no [SECTION] exists:
      - return one chunk with section_title=None and text=original text
    """
    if not text or not text.strip():
        return []

    if "[SECTION]" not in text and "[section]" not in text:
        return [{"section_title": None, "text": text.strip()}]

    blocks = [b.strip() for b in SECTION_SPLIT_PATTERN.split(text) if b.strip()]
    chunks: List[Dict[str, str]] = []

    for block in blocks:
        # block starts with "[SECTION] ..."
        # normalize whitespace
        block = re.sub(r"\s+", " ", block).strip()

        m = SECTION_HEADER_PATTERN.match(block)
        if not m:
            # 如果遇到异常格式，安全兜底：不丢内容
            chunks.append({"section_title": None, "text": block})
            continue

        section_id = m.group(1).strip()
        remainder = m.group(2).strip()  # header后面的内容，属于text的一部分

        chunks.append(
            {
                "section_title": section_id,
                "text": remainder
            }
        )

    return chunks


def process_records_by_section(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Read records one by one and split each by Section.
    Keeps all original fields except 'text', then adds:
      - id (new)
      - section_title
      - text (chunked)
    """
    output_records: List[Dict[str, Any]] = []

    for rec in records:
        original_text = rec.get("text", "")
        section_chunks = chunk_by_section(original_text)

        # Keep everything except text (we'll replace it)
        base_meta = {k: v for k, v in rec.items() if k != "text"}

        for idx, chunk in enumerate(section_chunks):
            new_rec = dict(base_meta)

            new_rec["section_title"] = chunk["section_title"]
            new_rec["text"] = chunk["text"]

            base_id = rec.get("id", "unknown")
            new_rec["id"] = f"{base_id}_section_{idx + 1}"

            output_records.append(new_rec)

    return output_records


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main(input_path: Union[str, Path], output_path: Union[str, Path]) -> None:
    input_path = Path(input_path)
    output_path = Path(output_path)

    data = load_json(input_path)
    if not isinstance(data, list):
        raise ValueError("Expected input JSON to be a list of records")

    processed = process_records_by_section(data)
    save_json(processed, output_path)


if __name__ == "__main__":
    main(
        input_path="../data/processed/act-2011-070/act-2011-070_chunked_division.json",
        output_path="../data/processed/act-2011-070/act-2011-070_chunked_section.json",
    )
