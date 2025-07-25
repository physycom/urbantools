#!/usr/bin/env python3
"""
Test script to validate the geometry merging improvements in merge_redundant_edges.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import networkx as nx
from shapely.geometry import LineString
from urbantools.cartopy.cartopy import merge_redundant_edges, merge_edge_attributes


def test_geometry_merging():
    """Test the improved geometry merging functionality."""

    print("Testing geometry merging improvements...")

    # Test 1: Simple sequential lines that should connect perfectly
    print("\n=== Test 1: Sequential connected lines ===")

    # Create two connected LineStrings
    line1 = LineString([(0, 0), (1, 0), (2, 0)])  # A->B
    line2 = LineString([(2, 0), (3, 0), (4, 0)])  # B->C

    edge1 = {
        "geometry": line1,
        "poly_length": line1.length,
        "length": line1.length,
        "name": "edge1",
        "maxspeed": 50,
        "mean_flow": 100,
        "width_TF": 3.5,
    }

    edge2 = {
        "geometry": line2,
        "poly_length": line2.length,
        "length": line2.length,
        "name": "edge2",
        "maxspeed": 50,
        "mean_flow": 80,
        "width_TF": 3.5,
    }

    merged = merge_edge_attributes(edge1, edge2)

    print(f"Original line 1 length: {line1.length:.2f}")
    print(f"Original line 2 length: {line2.length:.2f}")
    print(f"Expected merged length: {line1.length + line2.length:.2f}")
    print(f"Actual merged length: {merged['geometry'].length:.2f}")
    print(f"Merged geometry coordinates: {list(merged['geometry'].coords)}")

    # Test 2: Lines that need orientation correction
    print("\n=== Test 2: Lines requiring orientation correction ===")

    # Create lines where second line is reversed
    line1 = LineString([(0, 0), (1, 0), (2, 0)])  # A->B
    line2 = LineString([(4, 0), (3, 0), (2, 0)])  # C->B (reversed)

    edge1["geometry"] = line1
    edge1["poly_length"] = line1.length
    edge2["geometry"] = line2
    edge2["poly_length"] = line2.length

    merged = merge_edge_attributes(edge1, edge2)

    print(f"Original line 1: {list(line1.coords)}")
    print(f"Original line 2 (reversed): {list(line2.coords)}")
    print(f"Merged geometry coordinates: {list(merged['geometry'].coords)}")
    print(f"Expected merged length: {line1.length + line2.length:.2f}")
    print(f"Actual merged length: {merged['geometry'].length:.2f}")

    # Test 3: Graph-level test with merge_redundant_edges
    print("\n=== Test 3: Full graph merging test ===")

    G = nx.DiGraph()

    # Create a simple path: A -> B -> C
    # Where B is a degree-1 node that should be merged out

    # Add nodes
    G.add_node("A", x=0, y=0)
    G.add_node("B", x=2, y=0)  # This should be merged out
    G.add_node("C", x=4, y=0)

    # Add edges with geometry
    G.add_edge(
        "A",
        "B",
        geometry=LineString([(0, 0), (1, 0), (2, 0)]),
        poly_length=2.0,
        length=2.0,
        name="AB_road",
        maxspeed=50,
        mean_flow=100,
        width_TF=3.5,
    )

    G.add_edge(
        "B",
        "C",
        geometry=LineString([(2, 0), (3, 0), (4, 0)]),
        poly_length=2.0,
        length=2.0,
        name="BC_road",
        maxspeed=50,
        mean_flow=80,
        width_TF=3.5,
    )

    print(f"Original graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"Node B in_degree: {G.in_degree('B')}, out_degree: {G.out_degree('B')}")

    # Merge redundant edges
    merged_graph = merge_redundant_edges(G)

    print(
        f"Merged graph: {merged_graph.number_of_nodes()} nodes, {merged_graph.number_of_edges()} edges"
    )

    if merged_graph.has_edge("A", "C"):
        edge_data = merged_graph["A"]["C"]
        print(f"Merged A->C edge length: {edge_data.get('length', 'N/A')}")
        if "geometry" in edge_data:
            print(f"Merged geometry length: {edge_data['geometry'].length:.2f}")
            print(f"Merged geometry coordinates: {list(edge_data['geometry'].coords)}")
    else:
        print("Warning: Expected A->C edge not found!")
        print(f"Available edges: {list(merged_graph.edges())}")

    print("\nGeometry merging tests completed!")


if __name__ == "__main__":
    test_geometry_merging()
