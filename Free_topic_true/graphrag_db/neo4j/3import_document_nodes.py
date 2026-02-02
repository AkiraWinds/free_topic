import csv
from collections import defaultdict
from neo4j_client import Neo4jClient

NODES_CSV = "../../data/neo4j_data/Document_Structure_Graph/nodes.csv"
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

# ---------- 1. 按 label 分组 ----------
nodes_by_label = defaultdict(list)

with open(NODES_CSV, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        nodes_by_label[row["label"]].append(row)

# ---------- 2. 每个 label 单独导入 ----------
with Neo4jClient("flat2") as session:
    for label, rows in nodes_by_label.items():
        print(f"Importing label={label}, count={len(rows)}")

        for batch in chunked(rows, BATCH_SIZE):
            session.run(
                f"""
                UNWIND $rows AS row
                MERGE (n {{id: row.id}})
                SET n += apoc.map.removeKeys(row, ['id', 'label'])
                SET n:`{label}`
                """,
                rows=batch,
            )

print("✅ Nodes imported correctly by label")
