"""
This module contains functions to process and manipulate directed graphs using NetworkX.
It includes functions to merge redundant edges, extract a subgraph based on mean flow,
and reconnect a subgraph to the original graph.
"""

import logging

import heapq
import networkx as nx
from shapely.geometry import LineString
from shapely.ops import linemerge, snap, unary_union
from tqdm import tqdm


def merge_redundant_edges(graph: nx.DiGraph) -> nx.DiGraph:
    """
    Merges degree-1 and degree-2 nodes in a directed graph by combining edges and attributes.

    - Degree-1 case: in_degree==out_degree==1 -> collapse A->B->C into A->C
    - Degree-2 case: in_degree==out_degree==2 -> collapse both A->B->C and C->B->A into A->C and C->A

    Edge attributes merged: poly_length, width_TF (length-weighted mean), maxspeed (max),
    mean_flow, name/highway from the first segment, forbidden_turns from the second, and geometry.
    The geometry merge now snaps endpoints within a tolerance and unions both segments before merging.
    """

    def safe_merge_lines(
        line1: LineString, line2: LineString, tol: float = 1e-6
    ) -> LineString:
        """
        Merge two LineStrings by:
          1. Snapping line2 to line1 within `tol`.
          2. Taking their unary_union.
          3. Running linemerge to coalesce into a single LineString.
        """
        if line1 is None or line1.is_empty:
            return line2
        if line2 is None or line2.is_empty:
            return line1

        snapped = snap(line2, line1, tol)
        combined = unary_union([line1, snapped])

        merged = linemerge(combined)
        # If result has multiple parts (MultiLineString), pick the longest
        if merged.geom_type == "MultiLineString":
            parts = list(merged.geoms)
            merged = max(parts, key=lambda g: g.length)

        return merged

    G = graph.copy()
    changed = True

    while changed:
        changed = False
        candidates = [
            n
            for n in G.nodes()
            if (d := G.in_degree(n)) == G.out_degree(n) and d in (1, 2)
        ]

        for n in candidates:
            preds = list(G.predecessors(n))
            succs = list(G.successors(n))
            merged_edges = []

            for u in preds:
                for v in succs:
                    if u == v or not G.has_edge(u, n) or not G.has_edge(n, v):
                        continue

                    e1, e2 = G[u][n], G[n][v]
                    l1, l2 = e1.get("poly_length", 0), e2.get("poly_length", 0)
                    total_length = (l1 + l2) if (l1 and l2) else (l1 or l2 or 0)

                    w1 = e1.get("width_TF", e1.get("width_FT", 2))
                    w2 = e2.get("width_TF", e2.get("width_FT", 2))
                    width = (
                        (w1 * l1 + w2 * l2) / total_length
                        if total_length
                        else max(w1, w2)
                    )

                    maxspeed = max(e1.get("maxspeed", 0), e2.get("maxspeed", 0))
                    mean_flow = (e1.get("mean_flow", 0) + e2.get("mean_flow", 0)) / 2

                    geom1, geom2 = e1.get("geometry"), e2.get("geometry")
                    merged_geom = None
                    if geom1 or geom2:
                        merged_geom = safe_merge_lines(geom1, geom2)

                    def avg_list(a, b):
                        if not a and not b:
                            return [0] * 288
                        if a and b and len(a) == len(b):
                            return [(x + y) / 2 for x, y in zip(a, b)]
                        return a or b

                    merged_attrs = {
                        "poly_length": total_length,
                        "length": total_length,
                        "width_TF": width,
                        "maxspeed": maxspeed,
                        "mean_flow": mean_flow,
                        "geometry": merged_geom,
                        "name": e1.get("name", "") or e2.get("name", ""),
                        "highway": e1.get("highway", "") or e2.get("highway", ""),
                        "forbidden_turns": e2.get("forbidden_turns", ""),
                        "flow": avg_list(e1.get("flow"), e2.get("flow")),
                        "speed": avg_list(e1.get("speed"), e2.get("speed")),
                        "density": avg_list(e1.get("density"), e2.get("density")),
                    }
                    merged_edges.append((u, v, merged_attrs))

            if merged_edges:
                for u in preds:
                    if G.has_edge(u, n):
                        G.remove_edge(u, n)
                for v in succs:
                    if G.has_edge(n, v):
                        G.remove_edge(n, v)
                G.remove_node(n)
                for u, v, attrs in merged_edges:
                    G.add_edge(u, v, **attrs)
                changed = True

        logging.getLogger(__name__).info(
            f"Now: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges"
        )

    return G


def remove_dead_ends(graph: nx.DiGraph, type: str = "both") -> nx.DiGraph:
    """
    Remove dead ends from a directed graph. A dead end is defined as a node with no
    predecessors (in-degree 0) or no successors (out-degree 0). The function can
    remove dead ends in one or both directions, depending on the 'type' parameter.

    Parameters
    ----------
    graph (nx.DiGraph)
        The directed graph to process.
    type (str)
        The type of dead ends to remove. Can be 'in', 'out', or 'both'.
        - 'in': Remove nodes with no predecessors (in-degree 0).
        - 'out': Remove nodes with no successors (out-degree 0).
        - 'both': Remove nodes with no predecessors and no successors.

    Returns
    -------
    nx.DiGraph
        The processed directed graph with dead ends removed.

    Raises
    ------
    ValueError: If 'type' is not one of 'in', 'out', or 'both'.
    """
    pruned_graph = graph.copy()
    if type.lower() not in ["in", "out", "both"]:
        raise ValueError(f"Invalid type '{type}'. Must be 'in', 'out', or 'both'.")
    if type.lower() == "out" or type.lower() == "both":
        while True:
            dead_ends = [
                node
                for node in pruned_graph.nodes
                if pruned_graph.out_degree(node) == 0
            ]
            if not dead_ends:
                break
            pruned_graph.remove_nodes_from(dead_ends)
    if type.lower() == "in" or type.lower() == "both":
        while True:
            dead_ends = [
                node for node in pruned_graph.nodes if pruned_graph.in_degree(node) == 0
            ]
            if not dead_ends:
                break
            pruned_graph.remove_nodes_from(dead_ends)

    logging.getLogger(__name__).info(
        f"Removed {len(graph.nodes()) - len(pruned_graph.nodes())} dead ends from the graph."
    )

    return pruned_graph


def extract_subgraph(graph: nx.DiGraph, flow_fraction: float = 0.8) -> nx.DiGraph:
    """
    Extract a subgraph from the graph based on edge lengths and mean_flow.

    Parameters
    ----------
    graph (nx.DiGraph)
        The input directed graph.
    flow_fraction (float)
        The fraction of total mean_flow to consider for the subnetwork.

    Returns
    -------
    nx.DiGraph
        A subnetwork of the original graph.
    """
    # Calculate the crossing for each edge
    total_crossing = 0.0
    edge_crossings = []

    for u, v, attrs in graph.edges(data=True):
        if "length" not in attrs:
            raise ValueError(f"Edge {u}-{v} does not have 'length' attribute.")
        length = attrs["length"]
        if "mean_flow" not in attrs:
            raise ValueError(f"Edge {u}-{v} does not have 'flow' attribute.")
        mean_flow = attrs["mean_flow"]
        crossing = length * mean_flow
        edge_crossings.append((u, v, crossing, mean_flow))
        total_crossing += crossing

    crossing_limit = total_crossing * flow_fraction

    # Sort edges by mean_flow (descending)
    edge_crossings.sort(key=lambda x: x[3], reverse=True)

    cumulative = 0.0
    selected_edges = []

    # Select edges based on cumulative mean_flow fraction
    for u, v, crossing, _ in edge_crossings:
        cumulative += crossing
        if cumulative >= crossing_limit:
            break
        selected_edges.append((u, v))

    # Create a subgraph with the selected edges
    subgraph = graph.edge_subgraph(selected_edges).copy()

    logging.getLogger(__name__).info(
        f"Extracted subnetwork with {len(subgraph.nodes())} nodes and {len(subgraph.edges())} edges from the original graph with {len(graph.nodes())} nodes and {len(graph.edges())} edges."
    )

    return subgraph


def reconnect_subgraph(
    graph: nx.DiGraph, subgraph: nx.DiGraph, weight: str = "length"
) -> nx.DiGraph:
    """
    Reconnect a subgraph to the original graph by adding edges based on shortest paths.
    The idea of the algorithm is to find the shortest path (in the main graph) between
    nodes in the subgraph,and add those nodes/edges to the subgraph if they are not already
    present. This is done using a custom Dijkstra-like which stops as soon as it finds a
    node in the subgraph.

    Parameters
    ----------
    graph (nx.DiGraph)
        The original directed graph.
    subgraph (nx.DiGraph)
        The subgraph to reconnect.
    weight (str)
        The edge attribute to use as weight for the shortest path.

    Returns
    -------
    nx.DiGraph
        The reconnected subgraph.
    """
    n_nodes, n_edges = subgraph.number_of_nodes(), subgraph.number_of_edges()
    # Reconnect the subnetwork
    subgraph_nodes = set(subgraph.nodes())

    for source in tqdm(subgraph_nodes, desc="Reconnecting subnetwork", leave=False):
        visited = set()
        heap = [(0, source)]  # (cumulative_length, current_node)
        parents = {source: None}  # For reconstructing paths
        targets_remaining = subgraph_nodes - {source}

        while heap and targets_remaining:
            cum_length, current = heapq.heappop(heap)

            if current in visited:
                continue
            visited.add(current)

            if current in targets_remaining:
                # Reconstruct path backwards
                path = []
                node = current
                while node is not None:
                    path.append(node)
                    node = parents[node]
                path.reverse()

                # Add edges and nodes from path
                for i in range(len(path) - 1):
                    u, v = path[i], path[i + 1]
                    if not subgraph.has_edge(u, v):
                        if u not in subgraph:
                            subgraph.add_node(u, **graph.nodes[u])
                        if v not in subgraph:
                            subgraph.add_node(v, **graph.nodes[v])
                        subgraph.add_edge(u, v, **graph[u][v])

                targets_remaining.remove(current)
                continue  # Continue search for more targets

            # Expand neighbors
            for neighbor in graph.successors(current):
                if neighbor not in visited:
                    edge_data = graph[current][neighbor]
                    w = edge_data.get(weight, 1)
                    heapq.heappush(heap, (cum_length + w, neighbor))
                    if neighbor not in parents:  # Set parent if first time
                        parents[neighbor] = current

    logging.getLogger(__name__).info(
        f"Reconnected {subgraph.number_of_nodes() - n_nodes} nodes "
        f"and {subgraph.number_of_edges() - n_edges} edges to the subgraph."
    )

    return subgraph
