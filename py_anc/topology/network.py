from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

from .node import Node


@dataclass
class Network:
    nodes: Dict[int, Node] = field(default_factory=dict)

    def add_node(self, node: Node) -> None:
        self.nodes[node.node_id] = node

    def connect_nodes(self, node1_id: int, node2_id: int) -> None:
        node1 = self.nodes[node1_id]
        node2 = self.nodes[node2_id]

        if node2_id not in node1.neighbor_ids:
            node1.neighbor_ids.append(node2_id)
            node1.neighbor_ids.sort()
        if node1_id not in node2.neighbor_ids:
            node2.neighbor_ids.append(node1_id)
            node2.neighbor_ids.sort()
