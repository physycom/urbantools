"""
This module contains functions to manipulate traffic lights in a directed graph.
It includes functions to build Webster cycles based on traffic light data and saturation flow.
"""

import logging

import networkx as nx


def build_webster_cycles(
    graph: nx.DiGraph, traffic_lights: list, C: int = 5, lost_time_per_phase: float = 4
):
    """
    Build Webster cycles for a directed graph based on traffic lights and saturation flow.
    The function calculates the cycle time, green time, and red time for each traffic light
    in the graph. It also updates the graph with the calculated cycle times and green times.
    The function assumes that the graph has a 'flow' attribute for each edge.

    The Webster's formula is:
    c_0 = (1.5 * L + C) / (1 - Y)

    where:
        - c_0 is the cycle time
        - C is the base cycle time (default is 5)
        - L is the lost time per phase (default is 4)
        - Y is the sum of the ratios of flow to saturation flow for each incoming edge
    The green time for each traffic light is calculated as:
    g_i = (y_i / Y) * (c_0 - L)

    where:
        - g_i is the green time for traffic light i
        - y_i is the ratio of flow to saturation flow for traffic light i
        - Y is the sum of the ratios of flow to saturation flow for all incoming edges
        - c_0 is the cycle time
        - L is the lost time per phase

    Parameters
    ----------
    graph (nx.DiGraph)
        The directed graph to process.
    traffic_lights (list)
        A list of traffic light nodes in the graph.
    C (int)
        The C parameter for Webster's formula (default is 5).
    lost_time_per_phase (float)
        The lost time per phase (seconds) for the traffic lights.
    Returns
    -------
    nx.DiGraph
        The directed graph with updated cycle times and green times for each traffic light.
    """
    WIDTH_TO_SATURATION_FLOW = {
        2.33: 1000,
        2.66: 1200,
        # From now on, the values are from Webster's formula
        3: 1675,
        3.33: 1700,
        3.66: 1725,
        4: 1775,
        4.33: 1875,
        4.66: 2025,
        5: 2250,
        5.33: 2450,
    }

    def get_closest_saturation_flow(width):
        """Find the closest saturation flow from the lookup table."""
        if width is None:
            return list(WIDTH_TO_SATURATION_FLOW.values())[0]  # fallback default
        closest_width = min(WIDTH_TO_SATURATION_FLOW, key=lambda w: abs(w - width))
        return WIDTH_TO_SATURATION_FLOW[closest_width]

    records = []

    for tl in traffic_lights:
        in_edges = graph.in_edges(tl, data=True)
        if not in_edges:
            logging.getLogger(__name__).warning(
                f"Traffic light {tl} has no incoming edges."
            )
            continue

        flows = []
        saturation_flows = []
        for _, _, data in in_edges:
            flow = data.get("flow", 0)
            width = data.get("width", None)
            s_flow = get_closest_saturation_flow(width)
            flows.append(flow)
            saturation_flows.append(s_flow)
        L = lost_time_per_phase * len(flows)

        if sum(flows) == 0:
            logging.getLogger(__name__).warning(f"Traffic light {tl} has no flow.")
            continue

        # If width is None, use the first available width in dict, else use the closest value

        y_ratios = [flow / s for flow, s in zip(flows, saturation_flows)]
        Y = sum(y_ratios)

        if Y >= 1:
            logging.getLogger(__name__).warning(
                f"Traffic light {tl} is saturated (Y >= 1)."
            )
            continue

        C_0 = (1.5 * L + C) / (1 - Y)  # Webster's formula for cycle time

        for (u, _, _), y in zip(list(in_edges), y_ratios):
            green_time = (y / Y) * (C_0 - L) if y > 0 else 1  # * scaling_factor
            records.append(
                {
                    "id": tl,
                    "nF": u,
                    "totalCycleTime": round(C_0),
                    "greenTime": round(green_time),
                }
            )

    return records
