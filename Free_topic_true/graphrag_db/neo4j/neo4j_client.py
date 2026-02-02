# neo4j_client.py
from neo4j import GraphDatabase
from typing import Literal, Optional
import os
import dotenv

dotenv.load_dotenv()

KGType = Literal["flat", "hierarchical"]


class Neo4jClient:
    """
    Unified Neo4j client for Flat KG and Hierarchical KG.

    Usage:
        with Neo4jClient("flat") as session:
            session.run("MATCH (n) RETURN count(n)")
    """

    def __init__(
        self,
        kg_type: KGType,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
    ):
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.getenv("NEO4J_USER", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD")

        if not self.password:
            raise ValueError(
                "Neo4j password not provided. "
                "Set NEO4J_PASSWORD env var or pass password explicitly."
            )

        if kg_type == "flat":
            self.database = "flatkg"
        elif kg_type == "hierarchical":
            self.database = "hierarchicalkg"
        elif kg_type == "flat2":
            self.database = "flat2"
        else:
            raise ValueError("kg_type must be 'flat' or 'hierarchical' or 'flat2'")

        self._driver = GraphDatabase.driver(
            self.uri,
            auth=(self.user, self.password),
            max_connection_lifetime=3600,
            connection_timeout=30,
        )

        self._session = None

    # ---------- Context manager API ----------

    def __enter__(self):
        self._session = self._driver.session(database=self.database)
        return self._session

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._session:
            self._session.close()
            self._session = None
        self._driver.close()

    # ---------- Explicit API (optional) ----------

    def get_session(self):
        return self._driver.session(database=self.database)

    def close(self):
        self._driver.close()
