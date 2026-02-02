from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer

# -----------------------------
# 1.  initialize LLM/SLM
# -----------------------------
llm = ChatOpenAI(
    model="gpt-4o-mini",   
    temperature=0,
    openai_api_key='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',   # 从环境变量读取
)


# -----------------------------
# 2.  define node types
# -----------------------------
allowed_nodes = [
    "SUBJECT",
    "OBJECT",
    "CONSEQUENCE",
    "TEST",
    "PROBE",
    "EXCEPTION",
    "DEFINITION",
    "CLASS",
]


# -----------------------------
# 3.  define tuple-style relationships
# -----------------------------
allowed_relationships = [
    # Legal effect
    ("SUBJECT", "ENTITY_EMPOWERED_TO", "CONSEQUENCE"),
    ("SUBJECT", "ENTITY_REQUIRED_TO", "CONSEQUENCE"),
    ("CONSEQUENCE", "AFFECTS", "OBJECT"),

    # Tests / conditions
    ("TEST", "TEST_CONCERNS", "SUBJECT"),
    ("TEST", "TEST_CONCERNS", "OBJECT"),
    ("TEST", "TEST_CONCERNS", "PROBE"),
    ("TEST", "ENTITY_TESTED", "PROBE"),
    ("TEST", "CONDITIONAL_CONSEQUENCE", "CONSEQUENCE"),

    # Exceptions
    ("EXCEPTION", "EXCEPTION_APPLIES_TO", "TEST"),
    ("EXCEPTION", "EXCEPTION_APPLIES_TO", "CONSEQUENCE"),

    # Definitions & classes
    ("DEFINITION", "DEFINES", "SUBJECT"),
    ("DEFINITION", "DEFINES", "OBJECT"),
    ("DEFINITION", "DEFINES", "CONSEQUENCE"),
    ("DEFINITION", "DEFINES", "TEST"),

    ("CLASS", "ENTITY_HAS_PROPERTY", "SUBJECT"),
    ("CLASS", "ENTITY_HAS_PROPERTY", "OBJECT"),

    # Logical / discourse relations
    ("SUBJECT", "SAME_ENTITY", "SUBJECT"),
    ("OBJECT", "SAME_ENTITY", "OBJECT"),
    ("TEST", "SAME_ENTITY", "TEST"),

    ("TEST", "CONTINUATION", "TEST"),
    ("TEST", "FOLLOWED_BY", "TEST"),
    ("TEST", "AND", "TEST"),
    ("TEST", "OR", "TEST"),

    ("CLASS", "SAME_CLASS", "CLASS"),
]


# -----------------------------
# 4.  define additional instructions to LLM/SLM
# -----------------------------
additional_instructions = """
Node type definitions:
- SUBJECT: An entity that gains powers or restrictions under a law.
- OBJECT: An entity (noun phrase) affected by the subject under a law; typically faces more restrictions when a subject gains power.
- CONSEQUENCE: The specific power or restriction conferred by the law, typically attributed to the subject.
- TEST: An explicit condition applied to an entity (Subject, Object, or Probe) that determines when a law applies.
- PROBE: An entity to which a TEST is applied that is not a Subject or an Object.
- EXCEPTION: A corollary to a TEST that specifies when a law does not apply.
- DEFINITION: A span of text serving to clarify the ordinary meaning of a term used in the legal text.
- CLASS: A modifier that serves to disambiguate an entity from others (e.g., specifying a "trial court" judge vs. other judges).

Node properties:
- Use property key "name" to store the main surface form of the entity.
- Use property key "definition" for nodes, storing the explanatory text.
- Do not invent properties. Only use the allowed property keys.

Only extract relations that are explicitly stated in the text.
Do not invent legal effects, conditions, or entities.

Example:

Text:
"If a person holds a valid driver's license, the person may operate a motor vehicle.
However, if the person is under 18 years old, this permission does not apply.
For the purposes of this section, a 'driver' means a person authorized to operate a vehicle."

Expected interpretation:

- "a person" is a SUBJECT.
- "holds a valid driver's license" is a TEST applied to the SUBJECT.
- "may operate a motor vehicle" is a CONSEQUENCE attributed to the SUBJECT.
- "is under 18 years old" is an EXCEPTION that restricts the applicability of the CONSEQUENCE.
- "driver" is a DEFINITION, where definition = "a person authorized to operate a vehicle".

Expected relations to extract (illustrative):

(TEST) --TEST_CONCERNS--> (SUBJECT)
(SUBJECT) --ENTITY_EMPOWERED_TO--> (CONSEQUENCE)
(EXCEPTION) --EXCEPTION_APPLIES_TO--> (CONSEQUENCE)
(DEFINITION) --DEFINES--> (SUBJECT)
"""


# -----------------------------
# 5.  initialize Transformer
# -----------------------------
transformer = LLMGraphTransformer(
    llm=llm,
    allowed_nodes=allowed_nodes,
    allowed_relationships=allowed_relationships,
    strict_mode=True,
    node_properties=["name", "definition"],
    relationship_properties=False,
    additional_instructions=additional_instructions,
)


# -----------------------------
# 6.  test text (minimal experiment)
# -----------------------------
text = """
If a person holds a valid driver's license, the person may operate a motor vehicle.
However, if the person is under 18 years old, this permission does not apply.
For the purposes of this section, a "driver" means a person authorized to operate a vehicle.
"""


doc = Document(page_content=text)


# -----------------------------
# 7.  run triple extraction
# -----------------------------
graph_docs = transformer.convert_to_graph_documents([doc])

graph = graph_docs[0]

print("=== Nodes ===")
for node in graph.nodes:
    print(node)

print("\n=== Relationships ===")
for rel in graph.relationships:
    print(rel)
