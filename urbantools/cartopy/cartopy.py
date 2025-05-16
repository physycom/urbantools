"""
This module contains functions to process and manipulate directed graphs using NetworkX.
It includes functions to merge redundant edges, extract a subgraph based on mean flow,
and reconnect a subgraph to the original graph.
"""

import logging

import heapq
import networkx as nx
from shapely.geometry import LineString
from tqdm import tqdm


def merge_redundant_edges(graph: nx.DiGraph) -> nx.DiGraph:
    """
    Merges degree-1 and degree-2 nodes in a directed graph by combining edges and attributes.

    - Degree-1 case: in_degree==out_degree==1 → collapse A→B→C into A→C
    - Degree-2 case: in_degree==out_degree==2 → collapse both A→B→C and C→B→A into A→C and C→A

    Edge attributes merged: poly_length, width_TF (length-weighted mean), maxspeed (max),
    mean_flow, name/highway from the first segment, forbidden_turns from the second, and geometry.

    Parameters
    ----------
    graph (nx.DiGraph)
        The directed graph to process.

    Returns
    -------
    nx.DiGraph
        The processed directed graph with redundant edges merged.
    """
    graph = graph.copy()
    changed = True

    n_nodes, n_edges = graph.number_of_nodes(), graph.number_of_edges()

    while changed:
        changed = False
        # collect all nodes whose in‑degree == out‑degree == 1 or 2
        candidates = [
            n
            for n in graph.nodes()
            if (deg := graph.in_degree(n)) == graph.out_degree(n) and deg in (1, 2)
        ]

        for n in candidates:
            preds = list(graph.predecessors(n))
            succs = list(graph.successors(n))

            merged_edges = []
            for u in preds:
                for v in succs:
                    if u == v:
                        continue  # skip self‑loops
                    if not (graph.has_edge(u, n) and graph.has_edge(n, v)):
                        continue

                    edge_un = graph[u][n]
                    edge_nv = graph[n][v]

                    length_un, length_nv = edge_un.get("poly_length"), edge_nv.get(
                        "poly_length"
                    )
                    length = length_un + length_nv

                    width = (
                        edge_un.get("width_FT", 2) * length_un
                        + edge_nv.get("width_FT", 2) * length_nv
                    ) / length

                    maxspeed = max(
                        edge_un.get("maxspeed", 30), edge_nv.get("maxspeed", 30)
                    )
                    mean_flow = (
                        edge_un.get("mean_flow", 0) + edge_nv.get("mean_flow", 0)
                    ) / 2

                    geometry_un, geometry_nv = edge_un.get("geometry"), edge_nv.get(
                        "geometry"
                    )
                    if geometry_un and geometry_nv:
                        geom = LineString(
                            list(geometry_un.coords) + list(geometry_nv.coords)
                        )
                    else:
                        geom = geometry_un or geometry_nv

                    merge_names = lambda n1, n2: (
                        n2
                        if n1 and n2 and n1.strip() in n2.strip()
                        else (
                            n1
                            if n1 and n2 and n2.strip() in n1.strip()
                            else (
                                f"{n1.strip()} / {n2.strip()}"
                                if n1 and n2
                                else (n1 or n2 or "")
                            )
                        )
                    )

                    merged_edges.append(
                        (
                            u,
                            v,
                            {
                                "poly_length": length,
                                "width_TF": width,
                                "highway": "unknown",
                                "name": merge_names(
                                    edge_un.get("name", ""), edge_nv.get("name", "")
                                ),
                                "maxspeed": maxspeed,
                                "mean_flow": mean_flow,
                                "geometry": geom,
                                "forbidden_turns": edge_nv.get("forbidden_turns", ""),
                            },
                        )
                    )

            # only remove/add if we actually have something to merge
            if merged_edges:
                # remove old edges and the node
                for u in preds:
                    if graph.has_edge(u, n):
                        graph.remove_edge(u, n)
                for v in succs:
                    if graph.has_edge(n, v):
                        graph.remove_edge(n, v)
                graph.remove_node(n)

                # add the new merged edges
                for u, v, attrs in merged_edges:
                    graph.add_edge(u, v, **attrs)

                changed = True

        logging.getLogger(__name__).info(
            f"Reduced graph from {n_nodes} nodes and {n_edges} edges to {graph.number_of_nodes()}"
            f" nodes and {graph.number_of_edges()} edges."
        )

    return graph


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
