from neo4j_client import Neo4jClient

CONSTRAINTS = [
    # general id constraints
    "CREATE CONSTRAINT law_id IF NOT EXISTS FOR (n:Law) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT part_id IF NOT EXISTS FOR (n:Part) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT division_id IF NOT EXISTS FOR (n:Division) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT section_id IF NOT EXISTS FOR (n:Section) REQUIRE n.id IS UNIQUE",
]

with Neo4jClient("flat") as session:
    for c in CONSTRAINTS:
        session.run(c)

print("✅ Constraints created")
