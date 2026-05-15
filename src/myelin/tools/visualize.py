"""Visual knowledge graph export: Mermaid.js + D3.js force-directed JSON.

Exports Myelin's entity graph in two formats:
  - Mermaid.js:   graph LR / graph TD syntax for diagram-as-code
  - D3.js JSON:   nodes + links dict for force-directed rendering

Usage:
    viz = Visualizer(entity_store, graph)
    mermaid = viz.export_mermaid("hermes", depth=2)
    d3json  = viz.export_d3_json("hermes", depth=2)
"""

from __future__ import annotations

from typing import Any

from ..knowledge.entities import EntityStore
from ..knowledge.graph import KnowledgeGraph


def _safe_name(raw: str) -> str:
    """Sanitize a name for Mermaid node labels (no special chars)."""
    return raw.replace('"', "'").replace("[", "(").replace("]", ")").replace("{", "(").replace("}", ")")


class Visualizer:
    """Export the entity knowledge graph as Mermaid or D3.js JSON."""

    def __init__(self, entities: EntityStore, graph: KnowledgeGraph):
        self.entities = entities
        self.graph = graph

    # ── public entry points ────────────────────────────────────

    def export_mermaid(self, entity_name: str | None = None, depth: int = 2) -> str:
        """Return a Mermaid graph definition string.

        If *entity_name* is given, only the subgraph around that entity
        is exported.  Otherwise the top-50 entities and all their
        relationships are used.
        """
        nodes, edges = self._collect_graph(entity_name, depth)
        return self._render_mermaid(nodes, edges)

    def export_d3_json(self, entity_name: str | None = None, depth: int = 2) -> dict[str, Any]:
        """Return a D3.js force-directed graph JSON blob.

        Shape:
            {"nodes": [{"id": str, "group": str, "size": float}],
             "links": [{"source": str, "target": str, "relation": str, "strength": float}]}
        """
        nodes, edges = self._collect_graph(entity_name, depth)
        return self._render_d3_json(nodes, edges)

    # ── graph data collection ──────────────────────────────────

    def _collect_graph(
        self,
        entity_name: str | None,
        depth: int,
    ) -> tuple[dict[str, dict], list[dict[str, Any]]]:
        """Collect nodes (id -> info) and edges from the knowledge graph.

        Returns (nodes_dict, edges_list).
        """
        nodes: dict[str, dict] = {}
        edges: list[dict[str, Any]] = []

        if entity_name:
            # Find the entity and BFS around it
            found = self.entities.search(entity_name)
            if not found:
                return nodes, edges
            start_entity = found[0]
            start_id = start_entity["id"]
            self._add_node(nodes, start_entity)

            subgraph = self.graph.bfs_subgraph(
                start_entity_id=start_id,
                max_depth=depth,
                min_strength=0.0,
                max_nodes=200,
            )
            for n in subgraph.get("nodes", []):
                self._add_node(nodes, n)
            for e in subgraph.get("edges", []):
                self._add_edge(edges, e, nodes)
        else:
            # No entity: grab top 50 entities + all connections among them
            top = self.entities.get_top_entities(limit=50)
            for n in top:
                self._add_node(nodes, n)

            if nodes:
                eids = list(nodes.keys())
                placeholders = ",".join("?" * len(eids))
                # relationships where BOTH source AND target are in our set
                rels = self.graph.db.fetchall(
                    f"SELECT * FROM relationships "
                    f"WHERE source_entity_id IN ({placeholders}) "
                    f"AND target_entity_id IN ({placeholders}) "
                    f"ORDER BY strength DESC",
                    tuple(eids + eids),
                )
                for e in rels:
                    self._add_edge(edges, e, nodes)

        return nodes, edges

    def _add_node(self, nodes: dict, row: dict | Any) -> None:
        nid = str(row["id"])
        if nid not in nodes:
            mention_count = float(row.get("mention_count", 1))
            nodes[nid] = {
                "id": nid,
                "name": row.get("canonical_name", row.get("name", "?")),
                "entity_type": row.get("entity_type", "unknown"),
                "mention_count": mention_count,
                "size": max(2.0, min(10.0, mention_count * 0.5 + 2.0)),
            }

    def _add_edge(self, edges: list, row: dict | Any, nodes: dict) -> None:
        src = str(row["source_entity_id"])
        tgt = str(row["target_entity_id"])
        if src in nodes and tgt in nodes:
            edges.append({
                "source_id": src,
                "target_id": tgt,
                "relation_type": row.get("relation_type", "related_to"),
                "strength": float(row.get("strength", 1.0)),
            })

    # ── mermaid rendering ──────────────────────────────────────

    def _render_mermaid(self, nodes: dict[str, dict], edges: list[dict]) -> str:
        """Build a ``graph LR`` Mermaid definition string."""
        if not nodes:
            return "graph LR\n    %% No entities found."

        lines = ["graph LR"]

        # Sanitised node-id-to-label mapping
        node_labels: dict[str, str] = {}
        for nid, info in nodes.items():
            safe = _safe_name(info["name"])
            label = safe
            if nid in node_labels:
                continue
            node_labels[nid] = label
            lines.append(f'    {nid}["{label}"]')

        for e in edges:
            src = e["source_id"]
            tgt = e["target_id"]
            rel = _safe_name(e["relation_type"])
            if src in node_labels and tgt in node_labels:
                lines.append(f"    {src} -->|{rel}| {tgt}")

        return "\n".join(lines) + "\n"

    # ── D3.js JSON rendering ───────────────────────────────────

    def _render_d3_json(self, nodes: dict[str, dict], edges: list[dict]) -> dict[str, Any]:
        """Build a D3.js force-directed graph JSON object."""
        node_list = [
            {
                "id": info["name"],
                "group": info["entity_type"],
                "size": info["size"],
            }
            for info in nodes.values()
        ]

        link_list = [
            {
                "source": nodes[e["source_id"]]["name"],
                "target": nodes[e["target_id"]]["name"],
                "relation": e["relation_type"],
                "strength": e["strength"],
            }
            for e in edges
            if e["source_id"] in nodes and e["target_id"] in nodes
        ]

        return {"nodes": node_list, "links": link_list}
