import os
import json
from collections import defaultdict
from neo4j_client import Neo4jClient

JSON_DIR = "../../data/neo4j_data/Legal_Discourse_Graph/act-1997-078/"
BATCH_SIZE = 500


def chunked(iterable, size):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def load_relationships_from_json_dir(json_dir):
    """
    读取 JSON 文件中的 relationships
    - MENTIONS：单独存
    - 其他关系：ENTITY → ENTITY
    """
    mentions = []
    semantic_rels_by_type = defaultdict(list)

    for fname in os.listdir(json_dir):
        if not fname.endswith(".json"):
            continue

        path = os.path.join(json_dir, fname)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for rel in data.get("relationships", []):
            rel_type = rel.get("type")
            src = rel.get("source_id")
            tgt = rel.get("target_id")

            if not rel_type or not src or not tgt:
                continue

            if rel_type == "MENTIONS":
                mentions.append({
                    "section_id": src,
                    "entity_id": tgt
                })
            else:
                semantic_rels_by_type[rel_type].append({
                    "source_id": src,
                    "target_id": tgt
                })

    return mentions, semantic_rels_by_type


def import_mentions(session, mentions):
    """
    SECTION (文档层) → ENTITY
    """
    print(f"Importing MENTIONS, count={len(mentions)}")

    for batch in chunked(mentions, BATCH_SIZE):
        session.run(
            """
            UNWIND $rows AS row
            MATCH (s:ENTITY {id: row.section_id, discourse_role: "SECTION"})
            MATCH (e:ENTITY {id: row.entity_id})
            MERGE (s)-[:MENTIONS]->(e)
            """,
            rows=batch,
        )


def import_semantic_relationships(session, semantic_rels_by_type):
    """
    ENTITY → ENTITY
    """
    for rel_type, rows in semantic_rels_by_type.items():
        print(f"Importing {rel_type}, count={len(rows)}")

        for batch in chunked(rows, BATCH_SIZE):
            session.run(
                f"""
                UNWIND $rows AS row
                MATCH (a:ENTITY {{id: row.source_id}})
                MATCH (b:ENTITY {{id: row.target_id}})
                MERGE (a)-[:{rel_type}]->(b)
                """,
                rows=batch,
            )


def main():
    mentions, semantic_rels_by_type = load_relationships_from_json_dir(JSON_DIR)

    print(
        f"Total relationships loaded: "
        f"{len(mentions)} MENTIONS, "
        f"{sum(len(v) for v in semantic_rels_by_type.values())} semantic"
    )

    with Neo4jClient("flat2") as session:
        # 1️⃣ 先导入 MENTIONS（SECTION → ENTITY）
        import_mentions(session, mentions)

        # 2️⃣ 再导入语义关系（ENTITY → ENTITY）
        import_semantic_relationships(session, semantic_rels_by_type)

    print("✅ Relationships imported correctly (SECTION-safe)")


if __name__ == "__main__":
    main()
