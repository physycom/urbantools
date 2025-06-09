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


def set_logging_level(level: int = logging.INFO) -> None:
    """
    Set the logging level for the mod
    Parameters
    ----------
    level : int
        The logging level to set. Default is logging.INFO.
    """
    logging.getLogger(__name__).setLevel(level)


def merge_edge_attributes(e1: dict, e2: dict) -> dict:
    """
    Merge attributes from two edges into a single edge.

    Parameters
    ----------
    e1 : dict
        First edge attributes dictionary
    e2 : dict
        Second edge attributes dictionary

    Returns
    -------
    dict
        Merged edge attributes
    """

    def manipulate_list(a, b, optype="mean"):
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
            if optype == "mean":
                return ((arr_a + arr_b) / 2).tolist()
            elif optype == "sum":
                return (arr_a + arr_b).tolist()
        # If shapes don't match, return the first non-empty one
        return a

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

        try:
            snapped = snap(line2, line1, tol)
            combined = unary_union([line1, snapped])

            merged = linemerge(combined)
            # If result has multiple parts (MultiLineString), pick the longest
            if merged.geom_type == "MultiLineString":
                parts = list(merged.geoms)
                merged = max(parts, key=lambda g: g.length)

            return merged
        except (ValueError, Exception) as e:
            # If linemerge fails, return the longer of the two original lines
            logging.getLogger(__name__).warning(
                f"Failed to merge lines: {e}. Returning longer line."
            )
            return line1 if line1.length >= line2.length else line2

    # Calculate length-weighted attributes
    l1, l2 = e1.get("poly_length", 0), e2.get("poly_length", 0)
    total_length = (l1 + l2) if (l1 and l2) else (l1 or l2 or 0)

    w1 = e1.get("width_TF", e1.get("width_FT", 2))
    w2 = e2.get("width_TF", e2.get("width_FT", 2))
    width = (w1 * l1 + w2 * l2) / total_length if total_length else max(w1, w2)

    # Merge other attributes
    maxspeed = max(e1.get("maxspeed", 0), e2.get("maxspeed", 0))
    mean_flow = (e1.get("mean_flow", 0) + e2.get("mean_flow", 0)) / 2

    # Merge geometry
    geom1, geom2 = e1.get("geometry"), e2.get("geometry")
    merged_geom = None
    if geom1 or geom2:
        merged_geom = safe_merge_lines(geom1, geom2)

    # Merge list attributes
    speeds = np.array(manipulate_list(e1.get("speeds"), e2.get("speeds"), "mean"))
    counts = np.array(manipulate_list(e1.get("counts"), e2.get("counts"), "sum"))

    # Safe division to avoid division by zero
    densities = np.divide(
        counts * 12, speeds, out=np.zeros_like(counts, dtype=float), where=speeds != 0
    )

    logging.getLogger(__name__).info(
        f"Merged edges {e1.get('name', 'unknown')} and {e2.get('name', 'unknown')}"
    )

    return {
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
        "counts": counts,
        "flows": counts * 12,
        "speeds": speeds,
        "densities": densities,
    }


def merge_redundant_edges(graph: nx.DiGraph) -> nx.DiGraph:
    """
    Merges degree-1 and degree-2 nodes in a directed graph by combining edges and attributes.

    - Degree-1 case: in_degree==out_degree==1 -> collapse A->B->C into A->C
    - Degree-2 case: in_degree==out_degree==2 -> collapse both A->B->C and C->B->A into A->C and C->A

    Edge attributes merged: poly_length, width_TF (length-weighted mean), maxspeed (max),
    mean_flow, name/highway from the first segment, forbidden_turns from the second, and geometry.
    The geometry merge now snaps endpoints within a tolerance and unions both segments before merging.
    """

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
                    merged_attrs = merge_edge_attributes(e1, e2)
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
    selected_edges = set()

    # Select edges based on cumulative mean_flow fraction
    for u, v, crossing, _ in edge_crossings:
        cumulative += crossing
        if cumulative >= crossing_limit:
            break
        selected_edges.add((u, v))
        if graph.has_edge(v, u):
            selected_edges.add((v, u))

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


def handle_overlap(
    G: nx.DiGraph, strings_to_merge: list = None, strings_to_keep: list = []
) -> nx.DiGraph:
    """
    Handle overlapping edges by merging nodes that have edges containing specific strings.

    For each string in the list, the function:
    1. Cycles through all nodes in the graph
    2. Checks if a node has edges (incoming or outgoing) containing that string
    3. If a node has exactly two edges containing the string, removes the node and merges the edges
    4. If after processing, a node has only one edge left, removes it
    5. Otherwise, continues merging other qualifying nodes

    Parameters
    ----------
    G : nx.DiGraph
        The directed graph to process
    strings_to_merge : list, optional
        List of strings to search for in edge attributes. If None, returns the graph unchanged.

    Returns
    -------
    nx.DiGraph
        The processed graph with overlapping edges merged
    """
    if strings_to_merge is None:
        return G.copy()

    graph = G.copy()

    for search_string in strings_to_merge:
        logging.getLogger(__name__).info(f"Processing string: '{search_string}'")
        changed = True
        strings_to_not_merge = strings_to_keep + [
            s.lower() for s in strings_to_merge if s.lower() != search_string.lower()
        ]

        while changed:
            changed = False
            # Get a list of nodes to avoid dictionary changed during iteration
            nodes_to_check = list(graph.nodes())

            for node in nodes_to_check:
                if node not in graph:  # Node might have been removed
                    continue

                # Find edges containing the search string
                matching_edges = []

                # Check incoming edges
                for pred in graph.predecessors(node):
                    edge_data = graph[pred][node]
                    # Check various edge attributes for the string
                    for _, attr_value in edge_data.items():
                        if (
                            isinstance(attr_value, str)
                            and search_string.lower() in attr_value.lower()
                        ):
                            matching_edges.append(("in", pred, node, edge_data))
                            break

                # Check outgoing edges
                for succ in graph.successors(node):
                    edge_data = graph[node][succ]
                    # Check various edge attributes for the string
                    for _, attr_value in edge_data.items():
                        if (
                            isinstance(attr_value, str)
                            and search_string.lower() in attr_value.lower()
                        ):
                            matching_edges.append(("out", node, succ, edge_data))
                            break

                # If exactly two edges contain the string, merge them
                if len(matching_edges) == 2:
                    # Check if any other edges contain strings from strings_to_keep
                    skip_merge = False

                    if strings_to_not_merge:
                        # Check all edges connected to this node
                        all_edges = []
                        # Add incoming edges
                        for pred in graph.predecessors(node):
                            all_edges.append(graph[pred][node])
                        # Add outgoing edges
                        for succ in graph.successors(node):
                            all_edges.append(graph[node][succ])

                        if len(all_edges) < 4:
                            # Check if any edge contains strings to keep
                            for edge_data in all_edges:
                                for keep_string in strings_to_not_merge:
                                    for _, attr_value in edge_data.items():
                                        if (
                                            isinstance(attr_value, str)
                                            and keep_string.lower()
                                            in attr_value.lower()
                                        ):
                                            skip_merge = True
                                            logging.getLogger(__name__).debug(
                                                f"Skipping merge at node {node} because edge contains protected string '{keep_string}'"
                                            )
                                            break
                                    if skip_merge:
                                        break
                                if skip_merge:
                                    break

                    if skip_merge:
                        continue  # Skip this node and move to the next one

                    edge1_type, edge1_u, edge1_v, edge1_data = matching_edges[0]
                    edge2_type, edge2_u, edge2_v, edge2_data = matching_edges[1]

                    # Determine the new edge endpoints
                    if edge1_type == "in" and edge2_type == "out":
                        # pred -> node -> succ becomes pred -> succ
                        new_u, new_v = edge1_u, edge2_v
                        merged_attrs = merge_edge_attributes(edge1_data, edge2_data)
                    elif edge1_type == "out" and edge2_type == "in":
                        # pred -> node -> succ becomes pred -> succ
                        new_u, new_v = edge2_u, edge1_v
                        merged_attrs = merge_edge_attributes(edge2_data, edge1_data)
                    elif edge1_type == "in" and edge2_type == "in":
                        # Two incoming edges - merge them into one
                        # Keep the node, but merge the two incoming edges
                        # Remove both edges and add a merged edge from one predecessor
                        graph.remove_edge(edge1_u, edge1_v)
                        graph.remove_edge(edge2_u, edge2_v)
                        merged_attrs = merge_edge_attributes(edge1_data, edge2_data)
                        # Choose the predecessor with higher mean_flow or longer edge
                        if edge1_data.get("mean_flow", 0) >= edge2_data.get(
                            "mean_flow", 0
                        ):
                            graph.add_edge(edge1_u, node, **merged_attrs)
                        else:
                            graph.add_edge(edge2_u, node, **merged_attrs)
                        changed = True
                        continue
                    elif edge1_type == "out" and edge2_type == "out":
                        # Two outgoing edges - merge them into one
                        graph.remove_edge(edge1_u, edge1_v)
                        graph.remove_edge(edge2_u, edge2_v)
                        merged_attrs = merge_edge_attributes(edge1_data, edge2_data)
                        # Choose the successor with higher mean_flow or longer edge
                        if edge1_data.get("mean_flow", 0) >= edge2_data.get(
                            "mean_flow", 0
                        ):
                            graph.add_edge(node, edge1_v, **merged_attrs)
                        else:
                            graph.add_edge(node, edge2_v, **merged_attrs)
                        changed = True
                        continue
                    else:
                        continue

                    # Remove the original edges and node
                    graph.remove_edge(edge1_u, edge1_v)
                    graph.remove_edge(edge2_u, edge2_v)

                    # Check if node has any remaining edges
                    remaining_in = list(graph.predecessors(node))
                    remaining_out = list(graph.successors(node))

                    if len(remaining_in) + len(remaining_out) <= 1:
                        # Node has at most one edge left, remove it
                        if remaining_in:
                            logging.getLogger(__name__).info(
                                f"Removing edge '{graph[remaining_in[0]][node].get('name', 'unnamed')}' due to overlap with string '{search_string}'"
                            )
                            graph.remove_edge(remaining_in[0], node)
                            # Print name of the edge being removed

                        if remaining_out:
                            logging.getLogger(__name__).info(
                                f"Removing edge '{graph[node][remaining_out[0]].get('name', 'unnamed')}' due to overlap with string '{search_string}'"
                            )
                            graph.remove_edge(node, remaining_out[0])
                        if node in graph:
                            graph.remove_node(node)
                    else:
                        # Node has more edges, continue processing them
                        # Remove the node and connect remaining predecessors to successors
                        for pred in remaining_in:
                            for succ in remaining_out:
                                if pred != succ:
                                    pred_edge = graph[pred][node]
                                    succ_edge = graph[node][succ]
                                    merged_remaining = merge_edge_attributes(
                                        pred_edge, succ_edge
                                    )
                                    graph.add_edge(pred, succ, **merged_remaining)

                        # Remove all edges connected to the node
                        for pred in remaining_in:
                            graph.remove_edge(pred, node)
                        for succ in remaining_out:
                            graph.remove_edge(node, succ)
                        if node in graph:
                            graph.remove_node(node)

                    # Add the new merged edge
                    if new_u != new_v and new_u in graph and new_v in graph:
                        graph.add_edge(new_u, new_v, **merged_attrs)

                    changed = True
                    logging.getLogger(__name__).debug(
                        f"Merged edges at node {node} for string '{search_string}'"
                    )

                elif len(matching_edges) == 4:
                    logging.getLogger(__name__).info(
                        f"Node {node} has 4 edges containing '{search_string}'. Merging by continuity."
                    )
                    # Merge two pairs of edges without creating bidirectional connections

                    # Find pairs of edges that can be merged without creating bidirectional connections
                    edge_pairs = []
                    used_edges = set()

                    for i, (type1, u1, v1, data1) in enumerate(matching_edges):
                        if i in used_edges:
                            continue

                        best_match = None

                        for j, (type2, u2, v2, data2) in enumerate(
                            matching_edges[i + 1 :], i + 1
                        ):
                            if j in used_edges:
                                continue

                            # Check if edges can form a continuous path and avoid u,v with v,u pattern
                            can_connect = False
                            if type1 == "in" and type2 == "out":
                                # Check if this would create a bidirectional edge (u,v with v,u)
                                if not (u1 == v2 and v1 == u2):
                                    can_connect = True
                            elif type1 == "out" and type2 == "in":
                                # Check if this would create a bidirectional edge (u,v with v,u)
                                if not (u2 == v1 and v2 == u1):
                                    can_connect = True

                            if can_connect and best_match is None:
                                best_match = j
                                break  # Take the first valid connection

                        if best_match is not None:
                            edge_pairs.append((i, best_match))
                            used_edges.add(i)
                            used_edges.add(best_match)

                    # Merge the paired edges
                    merged_count = 0
                    for pair_idx, (i, j) in enumerate(edge_pairs):
                        type1, u1, v1, data1 = matching_edges[i]
                        type2, u2, v2, data2 = matching_edges[j]

                        # Determine new edge endpoints
                        if type1 == "in" and type2 == "out":
                            new_u, new_v = u1, v2
                            merged_attrs = merge_edge_attributes(data1, data2)
                        elif type1 == "out" and type2 == "in":
                            new_u, new_v = u2, v1
                            merged_attrs = merge_edge_attributes(data2, data1)
                        else:
                            continue

                        # Remove original edges
                        if graph.has_edge(u1, v1):
                            graph.remove_edge(u1, v1)
                        if graph.has_edge(u2, v2):
                            graph.remove_edge(u2, v2)

                        # Add merged edge
                        if new_u != new_v and new_u in graph and new_v in graph:
                            graph.add_edge(new_u, new_v, **merged_attrs)
                            merged_count += 1
                            logging.getLogger(__name__).debug(
                                f"Merged continuous edges {u1}->{v1} and {u2}->{v2} into {new_u}->{new_v}"
                            )

                    # Handle any remaining edges at the node
                    if merged_count > 0:
                        remaining_in = list(graph.predecessors(node))
                        remaining_out = list(graph.successors(node))

                        if len(remaining_in) + len(remaining_out) == 0:
                            # No remaining edges, remove the node
                            if node in graph:
                                graph.remove_node(node)
                        elif len(remaining_in) + len(remaining_out) <= 2:
                            # Few remaining edges, merge them if possible
                            for pred in remaining_in:
                                for succ in remaining_out:
                                    if pred != succ:
                                        pred_edge = graph[pred][node]
                                        succ_edge = graph[node][succ]
                                        merged_remaining = merge_edge_attributes(
                                            pred_edge, succ_edge
                                        )
                                        graph.add_edge(pred, succ, **merged_remaining)

                            # Remove all edges connected to the node
                            for pred in remaining_in:
                                if graph.has_edge(pred, node):
                                    graph.remove_edge(pred, node)
                            for succ in remaining_out:
                                if graph.has_edge(node, succ):
                                    graph.remove_edge(node, succ)
                            if node in graph:
                                graph.remove_node(node)

                        changed = True

    logging.getLogger(__name__).info(
        f"Overlap handling complete. Graph now has {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges"
    )

    return graph
