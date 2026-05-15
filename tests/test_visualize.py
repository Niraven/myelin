"""Test knowledge graph visualizer — Mermaid.js + D3.js JSON export."""

from myelin.tools.visualize import Visualizer


def _seed_graph(store, graph):
    """Seed a minimal test graph."""
    e1 = store.upsert_entity("hermes", "tool", "hermes")
    e2 = store.upsert_entity("cloudflared", "tool", "cloudflared")
    e3 = store.upsert_entity("obsidian", "tool", "obsidian")
    e4 = store.upsert_entity("myelin", "tool", "myelin")

    graph.add_relationship(e1, e2, "uses", strength=1.0)
    graph.add_relationship(e1, e3, "uses", strength=0.8)
    graph.add_relationship(e2, e1, "triggers", strength=0.9)
    graph.add_relationship(e3, e4, "related_to", strength=0.7)

    return e1, e2, e3, e4


class TestVisualizer:
    def test_export_mermaid_full_graph(self, entity_store, graph):
        _seed_graph(entity_store, graph)
        viz = Visualizer(entity_store, graph)
        result = viz.export_mermaid()
        assert result.startswith("graph LR")
        assert "hermes" in result
        assert "cloudflared" in result
        assert "-->|uses|" in result

    def test_export_mermaid_with_entity(self, entity_store, graph):
        e1, e2, e3, _ = _seed_graph(entity_store, graph)
        viz = Visualizer(entity_store, graph)
        result = viz.export_mermaid("hermes", depth=2)
        assert result.startswith("graph LR")
        assert "hermes" in result
        assert "obsidian" in result or "cloudflared" in result

    def test_export_mermaid_empty_db(self, entity_store, graph):
        viz = Visualizer(entity_store, graph)
        result = viz.export_mermaid("nonexistent")
        assert "%% No entities found" in result

    def test_export_d3_json_full_graph(self, entity_store, graph):
        _seed_graph(entity_store, graph)
        viz = Visualizer(entity_store, graph)
        result = viz.export_d3_json()
        assert "nodes" in result
        assert "links" in result
        names = {n["id"] for n in result["nodes"]}
        assert "hermes" in names
        assert "cloudflared" in names

    def test_export_d3_json_with_entity(self, entity_store, graph):
        _seed_graph(entity_store, graph)
        viz = Visualizer(entity_store, graph)
        result = viz.export_d3_json("hermes", depth=1)
        assert len(result["nodes"]) >= 1
        assert len(result["links"]) >= 0

    def test_d3_json_link_structure(self, entity_store, graph):
        _seed_graph(entity_store, graph)
        viz = Visualizer(entity_store, graph)
        result = viz.export_d3_json("hermes")
        for link in result["links"]:
            assert "source" in link
            assert "target" in link
            assert "relation" in link
            assert "strength" in link

    def test_d3_json_node_structure(self, entity_store, graph):
        _seed_graph(entity_store, graph)
        viz = Visualizer(entity_store, graph)
        result = viz.export_d3_json()
        for node in result["nodes"]:
            assert "id" in node
            assert "group" in node
            assert "size" in node
