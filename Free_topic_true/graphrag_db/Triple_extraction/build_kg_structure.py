import os
import json
import csv
from collections import OrderedDict

INPUT_DIR = "../../data/processed/chunked/"
OUTPUT_DIR = "../../data/neo4j_data/Document_Structure_Graph3/"

os.makedirs(OUTPUT_DIR, exist_ok=True)

NODES_FILE = os.path.join(OUTPUT_DIR, "nodes.csv")
RELS_FILE = os.path.join(OUTPUT_DIR, "relationships.csv")

# --------------------------------------------------
# Containers
# --------------------------------------------------
nodes = OrderedDict()
relationships = []

# --------------------------------------------------
# Helper functions
# --------------------------------------------------
def add_node(node_id, label, properties):
    if node_id not in nodes:
        nodes[node_id] = {
            "id": node_id,
            "label": label,
            **properties
        }

def add_rel(start_id, rel_type, end_id):
    relationships.append({
        "start_id": start_id,
        "type": rel_type,
        "end_id": end_id
    })

# --------------------------------------------------
# Main processing
# --------------------------------------------------
for filename in os.listdir(INPUT_DIR):
    if not filename.endswith(".json"):
        continue

    path = os.path.join(INPUT_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Each file is a list[section]
    for section in data:

        # -----------------
        # Law
        # -----------------
        law_id = section["version_id"]

        add_node(
            node_id=law_id,
            label="Law",
            properties={
                "citation": section.get("citation"),
                "jurisdiction": section.get("jurisdiction"),
                "source": section.get("source"),
                "url": section.get("url"),
                "date": section.get("date"),
                "type": section.get("type"),
                "mime": section.get("mime"),
                "when_scraped": section.get("when_scraped"),
            }
        )

        # -----------------
        # Part
        # -----------------
        part_title = section.get("part_title")
        if not part_title:
            # Rare case: no Part, skip directly
            continue

        # "Part 4 Local crime prevention" -> 4
        part_number = part_title.split()[1]

        part_id = f"{law_id}_part_{part_number}"

        add_node(
            node_id=part_id,
            label="Part",
            properties={
                "part_number": part_number,
                "title": part_title,
            }
        )

        add_rel(law_id, "HAS_PART", part_id)

        # -----------------
        # Division (optional)
        # -----------------
        division_title = section.get("division_title")

        if division_title:
            # "Division 3 Safer community compacts" -> 3
            division_number = division_title.split()[1]

            division_id = (
                f"{law_id}_part_{part_number}_division_{division_number}"
            )

            add_node(
                node_id=division_id,
                label="Division",
                properties={
                    "division_number": division_number,
                    "title": division_title,
                }
            )

            add_rel(part_id, "HAS_DIVISION", division_id)

            parent_for_section = division_id
        else:
            parent_for_section = part_id

        # -----------------
        # Section (clean semantic ID)
        # -----------------
        section_number = section.get("section_title")
        if not section_number:
            continue

        section_id = (
            f"{law_id}_part_{part_number}_section_{section_number}"
        )

        add_node(
            node_id=section_id,
            label="Section",
            properties={
                "section_number": section_number,
                "text": section.get("text"),
                # ⭐ provenance: keep original scraper ID
                "source_id": section.get("id"),
            }
        )

        add_rel(parent_for_section, "HAS_SECTION", section_id)

# --------------------------------------------------
# Write nodes.csv
# --------------------------------------------------
all_props = set()
for node in nodes.values():
    all_props.update(node.keys())

fieldnames = ["id", "label"] + sorted(all_props - {"id", "label"})

with open(NODES_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for node in nodes.values():
        writer.writerow(node)

# --------------------------------------------------
# Write relationships.csv
# --------------------------------------------------
with open(RELS_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["start_id", "type", "end_id"]
    )
    writer.writeheader()
    for rel in relationships:
        writer.writerow(rel)

print("✅ Document Structure Graph exported successfully (ID-clean version):")
print(f"- Nodes file: {NODES_FILE}")
print(f"- Relationships file: {RELS_FILE}")
