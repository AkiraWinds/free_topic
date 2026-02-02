import json
import shutil
from pathlib import Path

# =========================
# CONFIG
# =========================
BASE_DIR = Path("../../neo4j_data/Legal_Discourse_Graph/")           # 根目录
OUTPUT_DIR = Path("../results/top_bottom_10")   # 输出目录
LAW_ID = "act-1997-078"           # 当前分析的 law
MODELS = ["slm", "llm", "ollama"]
TOP_K = 10


def load_token_count(json_path: Path) -> int:
    """Read token_count from a section JSON file."""
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("meta", {}).get("token_count", 0)


def process_model(model_name: str):
    print(f"\nProcessing model: {model_name}")

    input_dir = BASE_DIR / model_name / LAW_ID
    if not input_dir.exists():
        print(f"  Skipped (directory not found): {input_dir}")
        return

    # read all json files
    records = []
    for json_file in input_dir.glob("*.json"):
        token_count = load_token_count(json_file)
        records.append((json_file, token_count))

    if not records:
        print("  No JSON files found.")
        return

    # sort by token count
    records.sort(key=lambda x: x[1])

    bottom_k = records[:TOP_K]
    top_k = records[-TOP_K:]

    # prepare output dirs
    bottom_dir = OUTPUT_DIR / model_name / LAW_ID / "bottom_10"
    top_dir = OUTPUT_DIR / model_name / LAW_ID / "top_10"
    bottom_dir.mkdir(parents=True, exist_ok=True)
    top_dir.mkdir(parents=True, exist_ok=True)

    # copy files
    for src, _ in bottom_k:
        shutil.copy(src, bottom_dir / src.name)

    for src, _ in top_k:
        shutil.copy(src, top_dir / src.name)

    print(f"  Copied {len(bottom_k)} bottom-token sections")
    print(f"  Copied {len(top_k)} top-token sections")


def main():
    for model in MODELS:
        process_model(model)

    print("\nDone. Case study files saved to:", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
