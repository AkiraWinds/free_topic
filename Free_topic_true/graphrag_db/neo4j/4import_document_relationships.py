import csv
from collections import defaultdict
from neo4j_client import Neo4jClient

RELS_CSV = "../../data/neo4j_data/Document_Structure_Graph/relationships.csv"
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

# ---------- 1. 读取并按 type 分组 ----------
rels_by_type = defaultdict(list)

with open(RELS_CSV, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        rels_by_type[row["type"]].append(row)

# ---------- 2. 按关系类型分别导入 ----------
with Neo4jClient("flat2") as session:
    for rel_type, rows in rels_by_type.items():
        print(f"Importing {rel_type}, count={len(rows)}")

        for batch in chunked(rows, BATCH_SIZE):
            session.run(
                f"""
                UNWIND $rows AS row
                MATCH (a {{id: row.start_id}})
                MATCH (b {{id: row.end_id}})
                MERGE (a)-[:{rel_type}]->(b)
                """,
                rows=batch,
            )

print("✅ Relationships imported correctly")
