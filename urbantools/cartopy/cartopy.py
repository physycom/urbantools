"""
This module contains functions to process and manipulate directed graphs using NetworkX.
It includes functions to merge redundant edges, extract a subgraph based on flow,
and reconnect a subgraph to the original graph.
"""

import logging

import heapq
import networkx as nx
from shapely.geometry import LineString
from tqdm import tqdm

_log = logging.getLogger(__name__)


def merge_redundant_edges(graph: nx.DiGraph) -> nx.DiGraph:
    """
    This function merges redundant edges in a directed graph.
    A redundant edge is defined as an edge which has exactly one predecessor and one successor,
    in one (degree 2) or both (degree 4) directions.
    The function merges these edges by creating a new edge that connects the predecessors
    and successors, and combines their attributes (length, width, max speed, etc.).
    The function also handles the geometry of the edges, creating a new geometry that represents
    the merged edge.

    Parameters
    ----------
    graph (nx.DiGraph): The directed graph to process.

    Returns
    -------
    graph (nx.DiGraph): The processed directed graph with degree 2 nodes merged.
    """
    # Find all nodes with degree 2
    graph = graph.copy()
    for degree in [1, 2]:
        nodes_to_remove = [
            node
            for node in graph.nodes()
            if graph.in_degree(node) == degree and graph.out_degree(node) == degree
        ]

        number_of_nodes = len(graph.nodes()) + 1
        while len(graph.nodes()) < number_of_nodes:
            number_of_nodes = len(graph.nodes())

            for node in nodes_to_remove[:]:
                # Get predecessors and successors
                predecessors = list(graph.predecessors(node))
                successors = list(graph.successors(node))

                # if len(predecessors) != 2 or len(successors) != 2:
                #     continue  # Just a safety check

                # Extract edge attributes (lengths in particular)
                edge_attrs = {}
                exit_flag = False
                for pred in predecessors:
                    for succ in successors:
                        if pred == succ:
                            # remove this node from nodes_to_remove
                            nodes_to_remove.remove(node)
                            exit_flag = True
                            break

                        if graph.has_edge(pred, node) and graph.has_edge(node, succ):
                            length1 = graph[pred][node].get("poly_length", 0)
                            length2 = graph[node][succ].get("poly_length", 0)

                            width1 = graph[pred][node].get("width_TF", 2)
                            width2 = graph[node][succ].get("width_TF", 2)
                            # Width mean is weighted by length
                            width = (width1 * length1 + width2 * length2) / (
                                length1 + length2
                            )

                            max_speed = max(
                                graph[pred][node].get("maxspeed", 30),
                                graph[node][succ].get("maxspeed", 30),
                            )

                            flow = (
                                graph[pred][node].get("flow", 0)
                                + graph[node][succ].get("flow", 0)
                            ) / 2

                            geometry = LineString(
                                list(
                                    graph[pred][node]
                                    .get("geometry", LineString())
                                    .coords
                                )
                                + list(
                                    graph[node][succ]
                                    .get("geometry", LineString())
                                    .coords
                                )
                            )

                            # Merge the edge attributes, summing the length
                            edge_attrs[(pred, succ)] = {
                                "poly_length": length1 + length2,
                                "width_TF": width,
                                "highway": graph[pred][node].get("highway", "unknown"),
                                "maxspeed": max_speed,
                                "name": graph[pred][node].get("name", "no_name"),
                                "geometry": geometry,
                                "forbidden_turns": graph[node][succ].get(
                                    "forbidden_turns", ""
                                ),  # How to manage removed nodes?
                                "flow": flow,
                            }

                    if exit_flag:
                        break

                # Add the merged edges
                for (pred, succ), attrs in edge_attrs.items():
                    graph.add_edge(pred, succ, **attrs)

            # Remove nodes
            graph.remove_nodes_from(nodes_to_remove)
            nodes_to_remove = [
                node
                for node in graph.nodes()
                if graph.in_degree(node) == degree and graph.out_degree(node) == degree
            ]

    return graph


def extract_subgraph(graph: nx.DiGraph, flow_fraction: float = 0.8) -> nx.DiGraph:
    """
    Extract a subgraph from the graph based on edge lengths and flow.

    Parameters
    ----------
    graph (nx.DiGraph): The input directed graph.
    flow_fraction (float): The fraction of total flow to consider for the subnetwork.

    Returns
    -------
    nx.DiGraph: A subnetwork of the original graph.
    """
    # Calculate the crossing for each edge
    total_crossing = 0.0
    edge_crossings = []

    for u, v, attrs in graph.edges(data=True):
        if "length" not in attrs:
            raise ValueError(f"Edge {u}-{v} does not have 'length' attribute.")
        length = attrs["length"]
        if "flow" not in attrs:
            raise ValueError(f"Edge {u}-{v} does not have 'flow' attribute.")
        flow = attrs["flow"]
        crossing = length * flow
        edge_crossings.append((u, v, crossing, flow))
        total_crossing += crossing

    crossing_limit = total_crossing * flow_fraction

    # Sort edges by flow (descending)
    edge_crossings.sort(key=lambda x: x[3], reverse=True)

    cumulative = 0.0
    selected_edges = []

    # Select edges based on cumulative flow fraction
    for u, v, crossing, _ in edge_crossings:
        cumulative += crossing
        if cumulative >= crossing_limit:
            break
        selected_edges.append((u, v))

    # Create a subgraph with the selected edges
    subgraph = graph.edge_subgraph(selected_edges).copy()

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
    graph (nx.DiGraph): The original directed graph.
    subgraph (nx.DiGraph): The subgraph to reconnect.
    weight (str): The edge attribute to use as weight for the shortest path.

    Returns
    -------
    nx.DiGraph: The reconnected subgraph.
    """
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

    return subgraph
