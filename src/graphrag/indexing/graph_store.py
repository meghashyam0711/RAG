"""Knowledge-graph stores.

* ``NetworkXGraphStore`` - zero-infrastructure default, persisted as JSON.
* ``Neo4jGraphStore``    - production store (set NEO4J_URI / NEO4J_PASSWORD;
  works with Memgraph too via the Bolt protocol).

Both expose the same small interface used by retrieval and the MCP tools.
"""

from __future__ import annotations

import json
import logging
import threading
from collections import deque
from pathlib import Path
from typing import Protocol

import networkx as nx

from ..models import Entity, Relation

log = logging.getLogger(__name__)


class GraphStore(Protocol):
    def upsert(self, entities: list[Entity], relations: list[Relation]) -> None: ...
    def delete_doc(self, doc_id: str) -> None: ...
    def get(self, entity_id: str) -> Entity | None: ...
    def all_entities(self) -> list[Entity]: ...
    def neighborhood(self, seeds: list[str], hops: int, limit: int = 200) -> tuple[dict[str, int], list[dict]]: ...
    def stats(self) -> dict: ...
    def close(self) -> None: ...


# ---------------------------------------------------------------------------
class NetworkXGraphStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self.g = nx.MultiDiGraph()
        if path.exists():
            data = json.loads(path.read_text())
            self.g = nx.node_link_graph(data, directed=True, multigraph=True, edges="edges")

    def _save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(nx.node_link_data(self.g, edges="edges")))
        tmp.replace(self.path)

    def upsert(self, entities: list[Entity], relations: list[Relation]) -> None:
        with self._lock:
            for e in entities:
                if self.g.has_node(e.id):
                    d = self.g.nodes[e.id]
                    d["aliases"] = sorted(set(d.get("aliases", [])) | set(e.aliases) | {e.name})
                    d["chunk_ids"] = sorted(set(d.get("chunk_ids", [])) | set(e.chunk_ids))
                    d["doc_ids"] = sorted(set(d.get("doc_ids", [])) | set(e.doc_ids))
                    if e.description and not d.get("description"):
                        d["description"] = e.description
                else:
                    self.g.add_node(e.id, **e.model_dump(exclude={"id"}))
            existing = {(u, v, d.get("type"), d.get("chunk_id")) for u, v, d in self.g.edges(data=True)}
            for r in relations:
                key = (r.source_id, r.target_id, r.type, r.chunk_id)
                if key not in existing and self.g.has_node(r.source_id) and self.g.has_node(r.target_id):
                    self.g.add_edge(r.source_id, r.target_id, type=r.type, chunk_id=r.chunk_id, evidence=r.evidence)
                    existing.add(key)
            self._compute_communities()
            self._save()

    def _compute_communities(self) -> None:
        if self.g.number_of_nodes() < 3:
            return
        try:
            comms = nx.community.louvain_communities(nx.Graph(self.g), seed=7)
        except Exception:  # pragma: no cover
            return
        for i, members in enumerate(comms):
            for n in members:
                self.g.nodes[n]["community"] = i

    def delete_doc(self, doc_id: str) -> None:
        with self._lock:
            prefix = f"{doc_id}::"
            self.g.remove_edges_from(
                [(u, v, k) for u, v, k, d in self.g.edges(keys=True, data=True) if str(d.get("chunk_id", "")).startswith(prefix)]
            )
            for n, d in list(self.g.nodes(data=True)):
                docs = [x for x in d.get("doc_ids", []) if x != doc_id]
                if not docs:
                    self.g.remove_node(n)
                else:
                    d["doc_ids"] = docs
                    d["chunk_ids"] = [c for c in d.get("chunk_ids", []) if not c.startswith(prefix)]
            self._save()

    def get(self, entity_id: str) -> Entity | None:
        if not self.g.has_node(entity_id):
            return None
        d = dict(self.g.nodes[entity_id])
        d.pop("community", None)
        return Entity(id=entity_id, **d)

    def community(self, entity_id: str) -> int | None:
        return self.g.nodes[entity_id].get("community") if self.g.has_node(entity_id) else None

    def all_entities(self) -> list[Entity]:
        return [e for n in self.g.nodes if (e := self.get(n))]

    def neighborhood(self, seeds: list[str], hops: int, limit: int = 200) -> tuple[dict[str, int], list[dict]]:
        """Undirected BFS up to `hops`. Returns ({node: distance}, edges)."""
        dist: dict[str, int] = {}
        q: deque[str] = deque()
        for s in seeds:
            if self.g.has_node(s):
                dist[s] = 0
                q.append(s)
        edges: list[dict] = []
        seen_edges = set()
        while q and len(dist) < limit:
            u = q.popleft()
            if dist[u] >= hops:
                continue
            nbrs = [(u, v, d) for _, v, d in self.g.out_edges(u, data=True)] + \
                   [(w, u, d) for w, _, d in self.g.in_edges(u, data=True)]
            for a, b, d in nbrs:
                other = b if a == u else a
                ek = (a, b, d.get("type"), d.get("chunk_id"))
                if ek not in seen_edges:
                    seen_edges.add(ek)
                    edges.append({"source": a, "target": b, "type": d.get("type"), "chunk_id": d.get("chunk_id"),
                                  "evidence": d.get("evidence", ""), "hop": dist[u] + 1})
                if other not in dist:
                    dist[other] = dist[u] + 1
                    q.append(other)
        return dist, edges

    def stats(self) -> dict:
        types: dict[str, int] = {}
        for _, d in self.g.nodes(data=True):
            types[d.get("type", "?")] = types.get(d.get("type", "?"), 0) + 1
        return {"backend": "networkx", "nodes": self.g.number_of_nodes(), "edges": self.g.number_of_edges(), "node_types": types}

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
class Neo4jGraphStore:
    """Neo4j / Memgraph store. Nodes: (:Entity {id, name, type, aliases, chunk_ids, doc_ids})."""

    def __init__(self, uri: str, user: str, password: str | None) -> None:
        from neo4j import GraphDatabase  # type: ignore

        self.driver = GraphDatabase.driver(uri, auth=(user, password or ""))
        with self.driver.session() as s:
            s.run("CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE")
            s.run("CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type)")

    def upsert(self, entities: list[Entity], relations: list[Relation]) -> None:
        ent_rows = [e.model_dump() for e in entities]
        rel_rows = [r.model_dump() for r in relations]
        with self.driver.session() as s:
            s.run(
                """
                UNWIND $rows AS r
                MERGE (e:Entity {id: r.id})
                ON CREATE SET e.name = r.name, e.type = r.type, e.description = r.description,
                              e.aliases = r.aliases, e.chunk_ids = r.chunk_ids, e.doc_ids = r.doc_ids
                ON MATCH SET  e.aliases   = apoc.coll.toSet(coalesce(e.aliases, []) + r.aliases + [r.name]),
                              e.chunk_ids = apoc.coll.toSet(coalesce(e.chunk_ids, []) + r.chunk_ids),
                              e.doc_ids   = apoc.coll.toSet(coalesce(e.doc_ids, []) + r.doc_ids)
                """ if self._has_apoc(s) else
                """
                UNWIND $rows AS r
                MERGE (e:Entity {id: r.id})
                ON CREATE SET e.name = r.name, e.type = r.type, e.description = r.description,
                              e.aliases = r.aliases, e.chunk_ids = r.chunk_ids, e.doc_ids = r.doc_ids
                ON MATCH SET  e.aliases   = [x IN coalesce(e.aliases, []) WHERE NOT x IN r.aliases] + r.aliases,
                              e.chunk_ids = [x IN coalesce(e.chunk_ids, []) WHERE NOT x IN r.chunk_ids] + r.chunk_ids,
                              e.doc_ids   = [x IN coalesce(e.doc_ids, []) WHERE NOT x IN r.doc_ids] + r.doc_ids
                """,
                rows=ent_rows,
            )
            s.run(
                """
                UNWIND $rows AS r
                MATCH (a:Entity {id: r.source_id}), (b:Entity {id: r.target_id})
                MERGE (a)-[x:REL {type: r.type, chunk_id: r.chunk_id}]->(b)
                SET x.evidence = r.evidence
                """,
                rows=rel_rows,
            )

    _apoc: bool | None = None

    def _has_apoc(self, session) -> bool:
        if self._apoc is None:
            try:
                session.run("RETURN apoc.version()").consume()
                self._apoc = True
            except Exception:
                self._apoc = False
        return self._apoc

    def delete_doc(self, doc_id: str) -> None:
        with self.driver.session() as s:
            s.run("MATCH ()-[x:REL]->() WHERE x.chunk_id STARTS WITH $p DELETE x", p=f"{doc_id}::")
            s.run(
                """
                MATCH (e:Entity) WHERE $d IN e.doc_ids
                SET e.doc_ids = [x IN e.doc_ids WHERE x <> $d],
                    e.chunk_ids = [c IN e.chunk_ids WHERE NOT c STARTS WITH $p]
                WITH e WHERE size(e.doc_ids) = 0 DETACH DELETE e
                """,
                d=doc_id, p=f"{doc_id}::",
            )

    def get(self, entity_id: str) -> Entity | None:
        with self.driver.session() as s:
            rec = s.run("MATCH (e:Entity {id: $id}) RETURN e", id=entity_id).single()
        return Entity(**dict(rec["e"])) if rec else None

    def all_entities(self) -> list[Entity]:
        with self.driver.session() as s:
            return [Entity(**dict(r["e"])) for r in s.run("MATCH (e:Entity) RETURN e")]

    def neighborhood(self, seeds: list[str], hops: int, limit: int = 200) -> tuple[dict[str, int], list[dict]]:
        hops = max(1, min(int(hops), 4))  # bounded traversal
        cypher = f"""
            MATCH (s:Entity) WHERE s.id IN $seeds
            MATCH p = (s)-[:REL*1..{hops}]-(n:Entity)
            WITH p, length(p) AS d ORDER BY d LIMIT $limit
            UNWIND range(0, length(p)-1) AS i
            WITH p, i, relationships(p)[i] AS r
            RETURN startNode(r).id AS source, endNode(r).id AS target, r.type AS type,
                   r.chunk_id AS chunk_id, r.evidence AS evidence, i + 1 AS hop,
                   [x IN nodes(p) | x.id] AS path_nodes
        """
        dist: dict[str, int] = {s: 0 for s in seeds}
        edges, seen = [], set()
        with self.driver.session() as s:
            for rec in s.run(cypher, seeds=seeds, limit=limit):
                for i, nid in enumerate(rec["path_nodes"]):
                    dist[nid] = min(dist.get(nid, 99), i)
                key = (rec["source"], rec["target"], rec["type"], rec["chunk_id"])
                if key not in seen:
                    seen.add(key)
                    edges.append({k: rec[k] for k in ("source", "target", "type", "chunk_id", "evidence", "hop")})
        return dist, edges

    def community(self, entity_id: str) -> int | None:
        return None  # run Leiden via Neo4j GDS if desired: gds.leiden.write(...)

    def stats(self) -> dict:
        with self.driver.session() as s:
            n = s.run("MATCH (e:Entity) RETURN count(e) AS c").single()["c"]
            m = s.run("MATCH ()-[r:REL]->() RETURN count(r) AS c").single()["c"]
        return {"backend": "neo4j", "nodes": n, "edges": m}

    def close(self) -> None:
        self.driver.close()


def build_graph_store(settings) -> GraphStore:
    if settings.neo4j_uri:
        try:
            store = Neo4jGraphStore(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password)
            log.info("Graph store: Neo4j at %s", settings.neo4j_uri)
            return store
        except Exception as exc:
            log.warning("Neo4j unavailable (%s); falling back to NetworkX", exc)
    return NetworkXGraphStore(settings.graph_path)
