"""
This module contains functions to process and manipulate directed graphs using NetworkX.
It includes functions to merge redundant edges, extract a subgraph based on mean flow,
and reconnect a subgraph to the original graph.
"""

import logging

import heapq
import networkx as nx
import numpy as np
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
                        def is_empty(x):
                            if x is None:
                                return True
                            if isinstance(x, (list, tuple)):
                                return len(x) == 0
                            if hasattr(x, "size"):
                                return x.size == 0
                            return False

                        if is_empty(a) and is_empty(b):
                            return [0] * 288
                        if is_empty(a):
                            return b
                        if is_empty(b):
                            return a
                        # Both are non-empty, try to average them
                        arr_a = np.array(a)
                        arr_b = np.array(b)
                        if arr_a.shape == arr_b.shape:
                            return ((arr_a + arr_b) / 2).tolist()
                        # If shapes don't match, return the first non-empty one
                        return a

                    merged_attrs = {
                        "poly_length": total_length,
                        "length": total_length,
                        "nlanes": max(e1.get("nlanes", 1), e2.get("nlanes", 1)),
                        "width_TF": width,
                        "maxspeed": maxspeed,
                        "mean_flow": mean_flow,
                        "geometry": merged_geom,
                        "name": e1.get("name", "") or e2.get("name", ""),
                        "highway": e1.get("highway", "") or e2.get("highway", ""),
                        "forbidden_turns": e2.get("forbidden_turns", ""),
                        "flows": avg_list(e1.get("flows"), e2.get("flows")),
                        "speeds": avg_list(e1.get("speeds"), e2.get("speeds")),
                        "densities": avg_list(e1.get("densities"), e2.get("densities")),
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
            logging.getLogger(__name__).warning(
                f"Edge {u}-{v} does not have 'mean_flow' attribute. Assuming mean_flow=0."
            )
        mean_flow = attrs.get("mean_flow", 0)
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
    graph: nx.DiGraph, subgraph: nx.DiGraph, weight: str = "length", min_degree: int = 2, pruning: bool = True
) -> nx.DiGraph:
    """
    Reconnect a subgraph to the original graph using an incremental approach with pruning.
    This algorithm mimics the C++ FeatureSelection approach by:
    1. Starting with the subgraph as a base
    2. Incrementally adding edges from the original graph
    3. Pruning nodes with degree < min_degree if pruning is enabled
    4. Finding connected components and maintaining the largest one

    Parameters
    ----------
    graph (nx.DiGraph)
        The original directed graph.
    subgraph (nx.DiGraph)
        The subgraph to reconnect.
    weight (str)
        The edge attribute to use as weight for selecting edges.
    min_degree (int)
        Minimum degree threshold for pruning nodes.
    pruning (bool)
        Whether to enable pruning of low-degree nodes.

    Returns
    -------
    nx.DiGraph
        The reconnected subgraph.
    """
    n_nodes, n_edges = subgraph.number_of_nodes(), subgraph.number_of_edges()
    
    # Convert to undirected for degree calculations (mimicking C++ undirected graph)
    working_graph = subgraph.to_undirected().to_directed()
    subgraph_nodes = set(subgraph.nodes())
    
    # Get all edges from original graph, sorted by weight
    all_edges = []
    for u, v, attrs in graph.edges(data=True):
        edge_weight = attrs.get(weight, 1)
        all_edges.append((edge_weight, u, v, attrs))
    
    # Sort edges by weight (ascending - prefer shorter/lighter edges)
    all_edges.sort(key=lambda x: x[0])
    
    # Track removed nodes for pruning
    removed_nodes = set()
    
    def prune_low_degree_nodes():
        """Prune nodes with degree < min_degree"""
        if not pruning:
            return 0
            
        pruned_count = 0
        changed = True
        
        while changed:
            changed = False
            nodes_to_remove = []
            
            for node in working_graph.nodes():
                if node not in removed_nodes:
                    # Calculate total degree (in + out for directed graph)
                    total_degree = working_graph.in_degree(node) + working_graph.out_degree(node)
                    if total_degree < min_degree:
                        nodes_to_remove.append(node)
            
            if nodes_to_remove:
                removed_nodes.update(nodes_to_remove)
                pruned_count += len(nodes_to_remove)
                changed = True
                
        return pruned_count
    
    def get_largest_component():
        """Get the largest connected component from non-removed nodes"""
        # Create filtered graph without removed nodes
        active_nodes = [n for n in working_graph.nodes() if n not in removed_nodes]
        if not active_nodes:
            return nx.DiGraph()
            
        active_subgraph = working_graph.subgraph(active_nodes)
        
        # Find connected components in undirected version
        undirected_active = active_subgraph.to_undirected()
        components = list(nx.connected_components(undirected_active))
        
        if not components:
            return nx.DiGraph()
            
        # Get largest component
        largest_component = max(components, key=len)
        return working_graph.subgraph(largest_component).copy()
    
    # Incremental edge addition
    edges_added = 0
    
    for edge_weight, u, v, attrs in tqdm(all_edges, desc="Reconnecting subnetwork", leave=False):
        # Skip if edge already exists
        if working_graph.has_edge(u, v):
            continue
            
        # Add nodes if they don't exist
        if u not in working_graph:
            working_graph.add_node(u, **graph.nodes.get(u, {}))
        if v not in working_graph:
            working_graph.add_node(v, **graph.nodes.get(v, {}))
            
        # Add edge
        working_graph.add_edge(u, v, **attrs)
        edges_added += 1
        
        # Prune low-degree nodes
        pruned = prune_low_degree_nodes()
        
        # Check if we have a meaningful connected component
        current_active_nodes = len([n for n in working_graph.nodes() if n not in removed_nodes])
        
        if current_active_nodes >= len(subgraph_nodes):
            # Get largest component
            largest_comp = get_largest_component()
            
            # Check if largest component contains most of original subgraph nodes
            comp_nodes = set(largest_comp.nodes())
            overlap = len(subgraph_nodes.intersection(comp_nodes))
            
            if overlap >= len(subgraph_nodes) * 0.8:  # 80% overlap threshold
                logging.getLogger(__name__).info(
                    f"Reconnected with {largest_comp.number_of_nodes()} nodes "
                    f"and {largest_comp.number_of_edges()} edges. "
                    f"Added {edges_added} edges, pruned {len(removed_nodes)} nodes."
                )
                return largest_comp
    
    # Fallback: return largest component of current working graph
    final_graph = get_largest_component()
    
    if final_graph.number_of_nodes() == 0:
        logging.getLogger(__name__).warning("Reconnection failed, returning original subgraph")
        return subgraph
    
    logging.getLogger(__name__).info(
        f"Reconnected {final_graph.number_of_nodes() - n_nodes} nodes "
        f"and {final_graph.number_of_edges() - n_edges} edges to the subgraph."
    )

    return final_graph
