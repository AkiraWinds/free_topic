from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer

# -----------------------------
# 1. Initialize LLM / SLM
# -----------------------------
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    openai_api_key='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',   # 从环境变量读取
)

# -----------------------------
# 2. Define node types (UNIFIED)
# -----------------------------
# all nodes are ENTITY，discourse role for property 
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
# 4. Additional instructions to LLM
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

# -----------------------------
# 6. Test text
# -----------------------------
text = """
If a person holds a valid driver's license, the person may operate a motor vehicle.
However, if the person is under 18 years old, this permission does not apply.
For the purposes of this section, a "driver" means a person authorized to operate a vehicle.
"""

doc = Document(page_content=text)

# -----------------------------
# 7. Run triple extraction
# -----------------------------
graph_docs = transformer.convert_to_graph_documents([doc])
graph = graph_docs[0]

print("=== Nodes ===")
for node in graph.nodes:
    print(node)

print("\n=== Relationships ===")
for rel in graph.relationships:
    print(rel)
