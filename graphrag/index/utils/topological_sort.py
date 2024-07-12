# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""
Topological sort utility method.

kiku:
Topological sorting is an algorithm for ordering the nodes in a 
directed acyclic graph (DAG) such that for every directed edge from node A to node B, 
node A comes before node B in the ordering. This is often used in dependency resolution, 
task scheduling, and build systems.
"""

from graphlib import TopologicalSorter


def topological_sort(graph: dict[str, list[str]]) -> list[str]:
    """Topological sort."""
    ts = TopologicalSorter(graph)
    return list(ts.static_order())
