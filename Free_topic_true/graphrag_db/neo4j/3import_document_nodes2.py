import os
import json
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


def load_nodes_from_json_dir(json_dir):
    """
    遍历文件夹，读取所有 JSON 文件里的 nodes
    """
    all_nodes = []
    seen_ids = set()

    for fname in os.listdir(json_dir):
        if not fname.endswith(".json"):
            continue

        path = os.path.join(json_dir, fname)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for node in data.get("nodes", []):
            node_id = node.get("id")
            if not node_id:
                continue

            # 去重（非常重要）
            if node_id in seen_ids:
                continue

            seen_ids.add(node_id)

            all_nodes.append({
                "id": node_id,
                **(node.get("properties") or {})
            })

    return all_nodes


def main():
    nodes = load_nodes_from_json_dir(JSON_DIR)
    print(f"Total unique ENTITY nodes to import: {len(nodes)}")

    with Neo4jClient("flat2") as session:
        for batch in chunked(nodes, BATCH_SIZE):
            session.run(
                """
                UNWIND $rows AS row
                MERGE (n:ENTITY {id: row.id})
                SET n += apoc.map.removeKeys(row, ['id'])
                """,
                rows=batch,
            )

    print("✅ ENTITY nodes imported correctly from JSON files")


if __name__ == "__main__":
    main()
