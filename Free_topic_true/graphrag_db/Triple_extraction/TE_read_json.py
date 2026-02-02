'''
run: 
python3 TE_read_json.py
'''
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer
import json
import os
import hashlib
import re
import time
import tiktoken
from typing import List, Dict
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Enable verbose logging
os.environ["LANGCHAIN_TRACING_V2"] = "false"  # Disable LangSmith tracing
os.environ["LANGCHAIN_VERBOSE"] = "true"

# -----------------------------
# 1. Initialize LLM
# -----------------------------
llm = ChatOpenAI(
    model="gpt-4o-mini",   # Or your current model
    temperature=0,
    openai_api_key=os.getenv("OPENAI_API_KEY"),   # Read from environment variable
    timeout=120,  # Increase timeout to 120 seconds
    max_retries=3,  # Maximum 3 retries
    request_timeout=120,  # Request timeout
)


# -----------------------------
# 2. Define node types (UNIFIED)
# -----------------------------
# All entities unified as ENTITY, discourse role expressed via property
allowed_nodes = [
    "ENTITY",
]

# -----------------------------
# 3. Define relationships (role-based, not label-based)
# -----------------------------
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

    # Logical / discourse relations
    ("ENTITY", "SAME_ENTITY", "ENTITY"),
    ("ENTITY", "CONTINUATION", "ENTITY"),
    ("ENTITY", "FOLLOWED_BY", "ENTITY"),
    ("ENTITY", "AND", "ENTITY"),
    ("ENTITY", "OR", "ENTITY"),
    ("ENTITY", "SAME_CLASS", "ENTITY"),
]

# -----------------------------
# 4. Additional instructions to LLM (FULL ORIGINAL VERSION)
# -----------------------------
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


# -----------------------------
# 5. Initialize Transformer
# -----------------------------
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
# 6. Utility functions
# =========================================================

# Initialize tiktoken encoder (for gpt-4o-mini)
try:
    encoding = tiktoken.encoding_for_model("gpt-4o-mini")
except KeyError:
    # If model doesn't exist, use cl100k_base (GPT-4 encoding)
    encoding = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    """Count the number of tokens in text"""
    return len(encoding.encode(text))

def text_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def load_records(json_path: str) -> List[Dict]:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)
    
def sanitize_filename(name: str) -> str:
    return re.sub(r'[<>:"/\\|?*]', "_", name)


def make_section_node(section_id: str) -> Dict:
    """
    Hard-coded SECTION node (not generated by LLM)
    """
    return {
        "id": section_id,
        "type": "ENTITY",
        "properties": {
            "name": section_id,
            "discourse_role": "SECTION"
        }
    }


def graph_document_to_dict(graph_doc, record_id: str, text: str) -> Dict:
    """
    Convert GraphDocument to serializable dict
    + inject SECTION node
    + inject SECTION --MENTIONS--> ENTITY edges
    + add text, token count, and graph statistics to metadata
    """

    nodes = []
    relationships = []

    # ---------- 1. Section node ----------
    section_node = make_section_node(record_id)
    nodes.append(section_node)

    # ---------- 2. LLM-extracted entity nodes ----------
    for node in graph_doc.nodes:
        nodes.append({
            "id": node.id,
            "type": node.type,
            "properties": node.properties or {}
        })

    # ---------- 3. LLM-extracted semantic relationships ----------
    for rel in graph_doc.relationships:
        relationships.append({
            "source_id": rel.source.id,
            "source_type": rel.source.type,
            "type": rel.type,
            "target_id": rel.target.id,
            "target_type": rel.target.type,
            "properties": rel.properties or {}
        })

    # ---------- 4. Hard-coded MENTIONS relationships ----------
    for node in graph_doc.nodes:
        relationships.append({
            "source_id": record_id,     # SECTION
            "source_type": "ENTITY",
            "type": "MENTIONS",
            "target_id": node.id,       # extracted ENTITY
            "target_type": "ENTITY",
            "properties": {}
        })

    # ---------- 5. Calculate statistics ----------
    token_count = count_tokens(text)
    # Count different types of nodes and relationships
    llm_extracted_nodes = len(graph_doc.nodes)  # LLM extracted nodes (excluding SECTION)
    llm_extracted_rels = len(graph_doc.relationships)  # LLM extracted relationships (excluding MENTIONS)
    total_nodes = len(nodes)  # Including SECTION node
    total_rels = len(relationships)  # All relationships

    return {
        "meta": {
            "record_id": record_id,
            "text_hash": text_hash(text),
            "text": text,
            "token_count": token_count,
            "char_count": len(text),
            # New: graph statistics
            "graph_stats": {
                "total_nodes": total_nodes,  # Total nodes (including SECTION)
                "llm_extracted_nodes": llm_extracted_nodes,  # LLM extracted entity nodes
                "total_relationships": total_rels,  # Total relationships
                "llm_extracted_relationships": llm_extracted_rels,  # LLM extracted semantic relationships
                "mentions_relationships": total_rels - llm_extracted_rels  # MENTIONS relationships count
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
# 7. Main pipeline with BATCH PROCESSING + RESUME
# =========================================================
def main():
    input_path = "../../../data/processed/chunked/act-1997-078_chunked_section.json"
    output_dir = "../../../data/neo4j_data/Legal_Discourse_Graph/slm/act-1997-078"
    progress_file = "./extracted_graphs/act-1997-078_progress_slm.json"  # Track progress

    # Batch configuration
    BATCH_SIZE = 3  # Process 3 records per batch, then rest
    SLEEP_BETWEEN_BATCHES = 1  # Sleep 1 second between batches

    print(f"Loading records from {input_path}...")
    records = load_records(input_path)
    print(f"✓ Loaded {len(records)} records")

    # Load progress (resume mode)
    processed_ids = set()
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            progress = json.load(f)
            processed_ids = set(progress.get('processed_ids', []))
        print(f"📌 Resume mode: {len(processed_ids)} records already processed")
    
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
            print(f"⚠ Skipping record {idx}: missing id or text")
            skipped += 1
            continue

        # Skip already processed records
        if record_id in processed_ids:
            print(f"[{idx}/{total}] ⏭ Skipping (already processed): {record_id}")
            skipped += 1
            continue

        print(f"\n[{idx}/{total}] Processing: {record_id}")
        token_count = count_tokens(text)
        print(f"  Text: {len(text)} chars, {token_count} tokens")
        
        try:
            start_time = time.time()
            print(f"  → Calling LLM for graph extraction...")
            print(f"     [DEBUG] Creating Document object...")
            doc = Document(page_content=text)
            print(f"     [DEBUG] Document created, calling transformer...")
            
            graph_doc = transformer.convert_to_graph_documents([doc])[0]
            
            elapsed = time.time() - start_time
            print(f"     [DEBUG] Transformer returned successfully!")
            
            print(f"  → Extracted {len(graph_doc.nodes)} nodes, {len(graph_doc.relationships)} relationships (took {elapsed:.1f}s)")

            result = graph_document_to_dict(
                graph_doc=graph_doc,
                record_id=record_id,
                text=text
            )

            save_record(result, output_dir)
            # Display graph statistics
            stats = result["meta"]["graph_stats"]
            print(f"  → Graph: {stats['llm_extracted_nodes']} nodes, {stats['llm_extracted_relationships']} relationships (LLM)")
            print(f"           {stats['total_nodes']} total nodes, {stats['total_relationships']} total edges (with SECTION+MENTIONS)")
            print(f"  ✔ Saved to {output_dir}/record_{sanitize_filename(record_id)}.json")
            
            # Accumulate statistics
            processed_ids.add(record_id)
            processed += 1
            total_tokens += result["meta"]["token_count"]
            total_nodes_extracted += stats['llm_extracted_nodes']
            total_rels_extracted += stats['llm_extracted_relationships']
       
            # Save progress
            os.makedirs(os.path.dirname(progress_file), exist_ok=True)
            with open(progress_file, 'w') as f:
                json.dump({'processed_ids': list(processed_ids)}, f)
            
            # Rest between batches
            if processed % BATCH_SIZE == 0:
                print(f"\n  💤 Batch completed ({processed} records), sleeping {SLEEP_BETWEEN_BATCHES}s...")
                time.sleep(SLEEP_BETWEEN_BATCHES)
            
        except Exception as e:
            print(f"  ✗ Error processing {record_id}: {e}")
            failed += 1
            continue
    
    print(f"\n{'='*60}")
    print(f"✓ Completed!")
    print(f"  Total records: {total}")
    print(f"  Processed: {processed}")
    print(f"  Skipped: {skipped}")
    print(f"  Failed: {failed}")
    print(f"  Total tokens processed: {total_tokens:,}")
    if processed > 0:
        print(f"  Average tokens per record: {total_tokens / processed:.1f}")
        print(f"  Total nodes extracted: {total_nodes_extracted:,}")
        print(f"  Total relationships extracted: {total_rels_extracted:,}")
        print(f"  Average nodes per record: {total_nodes_extracted / processed:.1f}")
        print(f"  Average relationships per record: {total_rels_extracted / processed:.1f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()