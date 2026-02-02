"""
run:
python3 TE_read_json_ollama.py
"""

# =========================================================
# Imports
# =========================================================
import json
import os
import hashlib
import re
import time
from typing import List, Dict

from langchain_core.documents import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_ollama import ChatOllama

import tiktoken


# =========================================================
# 1. Initialize LOCAL LLM (Ollama)
# =========================================================
llm = ChatOllama(
    model="llama3.1:8b-instruct-q4_K_M",
    temperature=0,
    base_url="http://localhost:11434",
)

# =========================================================
# 2. Token counter (third-party, decoupled from LLM)
# =========================================================
# This is ONLY for statistics / reporting, not for inference.
try:
    encoding = tiktoken.encoding_for_model("gpt-4o-mini")
except KeyError:
    encoding = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    return len(encoding.encode(text))


# =========================================================
# 3. Graph schema (UNIFIED ENTITY)
# =========================================================
allowed_nodes = ["ENTITY"]

allowed_relationships = [
    # Legal effects
    ("ENTITY", "ENTITY_EMPOWERED_TO", "ENTITY"),
    ("ENTITY", "ENTITY_REQUIRED_TO", "ENTITY"),
    ("ENTITY", "AFFECTS", "ENTITY"),

    # Tests / conditions
    ("ENTITY", "TEST_CONCERNS", "ENTITY"),
    ("ENTITY", "ENTITY_TESTED", "ENTITY"),
    ("ENTITY", "CONDITIONAL_CONSEQUENCE", "ENTITY"),

    # Exceptions
    ("ENTITY", "EXCEPTS", "ENTITY"),

    # Definitions & classes
    ("ENTITY", "DEFINES", "ENTITY"),
    ("ENTITY", "ENTITY_HAS_PROPERTY", "ENTITY"),

    # Logical / discourse
    ("ENTITY", "SAME_ENTITY", "ENTITY"),
    ("ENTITY", "CONTINUATION", "ENTITY"),
    ("ENTITY", "FOLLOWED_BY", "ENTITY"),
    ("ENTITY", "AND", "ENTITY"),
    ("ENTITY", "OR", "ENTITY"),
    ("ENTITY", "SAME_CLASS", "ENTITY"),
]


# =========================================================
# 4. Prompt instructions (UNCHANGED)
# =========================================================
additional_instructions = """
All extracted nodes MUST have the label ENTITY.

Each ENTITY node MUST include a property called "discourse_role",
whose value MUST be exactly one of the following:

SUBJECT, OBJECT, CONSEQUENCE, TEST, PROBE, EXCEPTION, DEFINITION, CLASS

Node discourse role definitions:
- SUBJECT: An entity that gains powers or restrictions under a law.
- OBJECT: An entity (noun phrase) affected by the subject under a law; typically faces more restrictions when a subject gains power.
- CONSEQUENCE: The specific power or restriction conferred by the law, typically attributed to the subject.
- TEST: An explicit condition applied to an entity (Subject, Object, or Probe) that determines when a law applies.
- PROBE: An entity to which a TEST is applied that is not a Subject or an Object.
- EXCEPTION: A corollary to a TEST that specifies when a law does not apply.
- DEFINITION: A span of text serving to clarify the ordinary meaning of a term used in the legal text.
- CLASS: A modifier that serves to disambiguate an entity from others (e.g., specifying a "trial court" judge vs. other judges).


Discourse roles MUST NOT be encoded as node labels.
They must be encoded ONLY via the "discourse_role" property.

Node property requirements:
- Use property key "name" to store the main surface form of the entity.
- Use property key "definition" ONLY for DEFINITION entities, storing the explanatory text.
- Use property key "discourse_role" for the discourse role.
- Do not invent additional properties.

Only extract relations that are explicitly stated in the text.
Do NOT invent legal effects, conditions, or entities.

Relationship semantics MUST be inferred from:
- The discourse roles of the connected entities
- Explicit legal language in the text (e.g., shall, may, if, unless, except, means)

Example:

Text:
"If a person holds a valid driver's license, the person may operate a motor vehicle.
However, if the person is under 18 years old, this permission does not apply.
For the purposes of this section, a 'driver' means a person authorized to operate a vehicle."

Expected entities (illustrative):
- ENTITY(name="a person", discourse_role="SUBJECT")
- ENTITY(name="holds a valid driver's license", discourse_role="TEST")
- ENTITY(name="may operate a motor vehicle", discourse_role="CONSEQUENCE")
- ENTITY(name="is under 18 years old", discourse_role="EXCEPTION")
- ENTITY(name="driver", discourse_role="DEFINITION", definition="a person authorized to operate a vehicle")

Expected relations (illustrative):
(TEST) --TEST_CONCERNS--> (SUBJECT)
(SUBJECT) --ENTITY_EMPOWERED_TO--> (CONSEQUENCE)
(EXCEPTION) --EXCEPTION_APPLIES_TO--> (CONSEQUENCE)
(DEFINITION) --DEFINES--> (SUBJECT)
"""


# =========================================================
# 5. Graph Transformer
# =========================================================
transformer = LLMGraphTransformer(
    llm=llm,
    allowed_nodes=allowed_nodes,
    allowed_relationships=allowed_relationships,
    strict_mode=True,
    node_properties=["name", "definition", "discourse_role"],
    relationship_properties=False,
    additional_instructions=additional_instructions,
)


# =========================================================
# 6. Utilities
# =========================================================
def text_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def load_records(json_path: str) -> List[Dict]:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def sanitize_filename(name: str) -> str:
    return re.sub(r'[<>:"/\\|?*]', "_", name)

def make_section_node(section_id: str) -> Dict:
    return {
        "id": section_id,
        "type": "ENTITY",
        "properties": {
            "name": section_id,
            "discourse_role": "SECTION"
        }
    }


def graph_document_to_dict(graph_doc, record_id: str, text: str) -> Dict:
    nodes = []
    relationships = []

    # SECTION node
    nodes.append(make_section_node(record_id))

    # LLM nodes
    for node in graph_doc.nodes:
        nodes.append({
            "id": node.id,
            "type": node.type,
            "properties": node.properties or {}
        })

    # LLM relationships
    for rel in graph_doc.relationships:
        relationships.append({
            "source_id": rel.source.id,
            "source_type": rel.source.type,
            "type": rel.type,
            "target_id": rel.target.id,
            "target_type": rel.target.type,
            "properties": rel.properties or {}
        })

    # SECTION --MENTIONS--> ENTITY
    for node in graph_doc.nodes:
        relationships.append({
            "source_id": record_id,
            "source_type": "ENTITY",
            "type": "MENTIONS",
            "target_id": node.id,
            "target_type": "ENTITY",
            "properties": {}
        })

    token_count = count_tokens(text)

    return {
        "meta": {
            "record_id": record_id,
            "text_hash": text_hash(text),
            "text": text,
            "token_count": token_count,
            "char_count": len(text),
            "graph_stats": {
                "total_nodes": len(nodes),
                "llm_extracted_nodes": len(graph_doc.nodes),
                "total_relationships": len(relationships),
                "llm_extracted_relationships": len(graph_doc.relationships),
                "mentions_relationships": len(relationships) - len(graph_doc.relationships),
            }
        },
        "nodes": nodes,
        "relationships": relationships
    }


def save_record(result: Dict, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    safe_id = sanitize_filename(result["meta"]["record_id"])
    path = os.path.join(output_dir, f"record_{safe_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)


# =========================================================
# 7. Main pipeline (BATCH + RESUME)
# =========================================================
def main():
    input_path = "../../../data/processed/chunked/act-1997-078_chunked_section.json"
    output_dir = "../../../data/neo4j_data/Legal_Discourse_Graph/ollama/act-1997-078"
    progress_file = "./extracted_graphs/act-1997-078_progress_ollama.json"

    BATCH_SIZE = 3
    SLEEP_BETWEEN_BATCHES = 1

    records = load_records(input_path)

    processed_ids = set()
    if os.path.exists(progress_file):
        with open(progress_file, "r") as f:
            processed_ids = set(json.load(f).get("processed_ids", []))

    total = len(records)
    skipped = 0
    processed = 0
    failed = 0
    total_tokens = 0  # Track total tokens
    total_nodes_extracted = 0
    total_rels_extracted = 0

    for idx, rec in enumerate(records, 1):
        record_id = rec.get("id")
        text = rec.get("text")

        if not record_id or not text:
            skipped += 1
            continue

        if record_id in processed_ids:
            skipped += 1
            continue

        print(f"[{idx}/{len(records)}] Processing {record_id}")

        try:
            doc = Document(page_content=text)
            graph_doc = transformer.convert_to_graph_documents([doc])[0]

            result = graph_document_to_dict(graph_doc, record_id, text)
            save_record(result, output_dir)
            stats = result["meta"]["graph_stats"]

            processed_ids.add(record_id)
            processed += 1
            total_tokens += result["meta"]["token_count"]
  
            total_nodes_extracted += stats['llm_extracted_nodes']
            total_rels_extracted += stats['llm_extracted_relationships']

            os.makedirs(os.path.dirname(progress_file), exist_ok=True)
            with open(progress_file, "w") as f:
                json.dump({"processed_ids": list(processed_ids)}, f)

            if processed % BATCH_SIZE == 0:
                time.sleep(SLEEP_BETWEEN_BATCHES)

        except Exception as e:
            print(f"✗ Error processing {record_id}: {e}")
            failed += 1

    print("======================================")
    print(f"Processed: {processed}")
    print(f"Skipped:   {skipped}")
    print(f"Failed:    {failed}")
    print(f"Tokens:    {total_tokens:,}")
    print("======================================")
    if processed > 0:
        print(f"  Average tokens per record: {total_tokens / processed:.1f}")
        print(f"  Total nodes extracted: {total_nodes_extracted:,}")
        print(f"  Total relationships extracted: {total_rels_extracted:,}")
        print(f"  Average nodes per record: {total_nodes_extracted / processed:.1f}")
        print(f"  Average relationships per record: {total_rels_extracted / processed:.1f}")


if __name__ == "__main__":
    main()
