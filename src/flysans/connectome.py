from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SparseGraph:
    neuron_count: int
    indices: torch.Tensor
    weights: torch.Tensor
    body_ids: torch.Tensor | None = None
    edge_groups: torch.Tensor | None = None
    soma_positions: torch.Tensor | None = None
    anatomy_regions: torch.Tensor | None = None

    def validate(self) -> None:
        if self.indices.shape[0] != 2 or self.indices.shape[1] != self.weights.numel():
            raise ValueError("indices must be [2, edge_count] and match weights")
        if self.indices.numel() and (self.indices.min() < 0 or self.indices.max() >= self.neuron_count):
            raise ValueError("edge index outside neuron range")

    def to(self, device: torch.device | str) -> "SparseGraph":
        ids = None if self.body_ids is None else self.body_ids.to(device)
        groups = None if self.edge_groups is None else self.edge_groups.to(device)
        positions = None if self.soma_positions is None else self.soma_positions.to(device)
        regions = None if self.anatomy_regions is None else self.anatomy_regions.to(device)
        return SparseGraph(self.neuron_count, self.indices.to(device), self.weights.to(device), ids, groups, positions, regions)


def make_synthetic_graph(neuron_count: int = 256, edges_per_neuron: int = 8, seed: int = 0) -> SparseGraph:
    generator = torch.Generator().manual_seed(seed)
    edge_count = neuron_count * edges_per_neuron
    source = torch.randint(neuron_count, (edge_count,), generator=generator)
    target = torch.randint(neuron_count, (edge_count,), generator=generator)
    weights = torch.randn(edge_count, generator=generator) / (edges_per_neuron**0.5)
    graph = SparseGraph(neuron_count, torch.stack((target, source)), weights)
    graph.validate()
    return graph


def load_graph(path: str, device: torch.device | str = "cpu") -> SparseGraph:
    payload = torch.load(path, map_location=device, weights_only=True)
    graph = SparseGraph(int(payload["neuron_count"]), payload["indices"], payload["weights"], payload.get("body_ids"), payload.get("edge_groups"), payload.get("soma_positions"), payload.get("anatomy_regions"))
    graph.validate()
    return graph
