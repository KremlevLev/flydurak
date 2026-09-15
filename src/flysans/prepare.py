import argparse, json, urllib.request
from pathlib import Path
import numpy as np
import torch

BASE = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome"
FILES = {"annotations": "body-annotations-male-cns-v1.0-minconf-0.5.feather", "neurotransmitters": "body-neurotransmitters-male-cns-v1.0.feather", "edges": "connectome-weights-male-cns-v1.0-minconf-0.5.feather"}

def soma_positions(annotations, body_ids):
    body_col = _column(annotations, ("body", "bodyId", "body_id"))
    soma_col = _column(annotations, ("somaLocation", "soma_location", "soma"))
    mapping = {int(body): location for body, location in zip(annotations[body_col].to_numpy(), annotations[soma_col].to_pylist()) if location is not None and len(location) == 3}
    values = np.full((len(body_ids), 3), np.nan, dtype=np.float32)
    for index, body in enumerate(body_ids):
        if int(body) in mapping: values[index] = mapping[int(body)]
    return torch.from_numpy(values)

def anatomy_regions(annotations, body_ids):
    body_col = _column(annotations, ("body", "bodyId", "body_id"))
    class_col = _column(annotations, ("superclass", "class"))
    vnc_classes = {"vnc_intrinsic", "vnc_sensory", "vnc_motor", "vnc_efferent", "vnc_tbc", "vnc_sensory_tbc", "vnc_endocrine", "ascending_neuron", "sensory_ascending"}
    mapping = {int(body): int(str(group) in vnc_classes) for body, group in zip(annotations[body_col].to_numpy(), annotations[class_col].to_pylist())}
    return torch.tensor([mapping.get(int(body), 0) for body in body_ids], dtype=torch.uint8)

def download(destination: Path):
    destination.mkdir(parents=True, exist_ok=True)
    for name in FILES.values():
        target = destination / name
        if not target.exists():
            print(f"downloading {name}")
            urllib.request.urlretrieve(f"{BASE}/{name}", target)

def _column(table, candidates):
    lookup = {name.lower(): name for name in table.column_names}
    for candidate in candidates:
        if candidate.lower() in lookup: return lookup[candidate.lower()]
    raise ValueError(f"expected one of {candidates}; columns={table.column_names}")

def build(raw: Path, output: Path, minimum_weight: int):
    import pyarrow.feather as feather
    annotations = feather.read_table(raw / FILES["annotations"])
    body_col, status_col = _column(annotations, ("body", "bodyId", "body_id")), _column(annotations, ("status",))
    bodies, statuses = annotations[body_col].to_numpy(), annotations[status_col].to_numpy()
    body_ids = np.sort(bodies[np.asarray([str(v).lower() == "traced" for v in statuses])].astype(np.int64))
    edges = feather.read_table(raw / FILES["edges"])
    pre = edges[_column(edges, ("body_pre", "bodyId_pre", "body_pre_id"))].to_numpy().astype(np.int64)
    post = edges[_column(edges, ("body_post", "bodyId_post", "body_post_id"))].to_numpy().astype(np.int64)
    weight = edges[_column(edges, ("weight", "syn_count", "count"))].to_numpy().astype(np.float32)
    mask = (weight >= minimum_weight) & np.isin(pre, body_ids) & np.isin(post, body_ids)
    kept_pre = pre[mask]
    source, target, weight = np.searchsorted(body_ids, kept_pre), np.searchsorted(body_ids, post[mask]), np.log1p(weight[mask])
    nt = feather.read_table(raw / FILES["neurotransmitters"])
    nt_body = nt[_column(nt, ("body", "bodyId", "body_id"))].to_numpy().astype(np.int64)
    nt_value = nt[_column(nt, ("predictedNt", "predicted_nt", "neurotransmitter"))].to_numpy()
    nt_map = {int(body): str(value).lower() for body, value in zip(nt_body, nt_value)}
    labels = np.asarray([nt_map.get(int(body), "unknown") for body in kept_pre])
    groups = np.full(len(labels), 2, dtype=np.int64)
    groups[np.isin(labels, ("acetylcholine", "ach"))] = 0
    groups[np.isin(labels, ("gaba", "glutamate", "histamine"))] = 1
    weight[groups == 1] *= -1.0
    incoming = np.bincount(target, weights=np.abs(weight), minlength=len(body_ids))
    weight /= np.maximum(incoming[target], 1.0)
    counts = {"excitatory": int((groups == 0).sum()), "inhibitory": int((groups == 1).sum()), "modulatory_or_unknown": int((groups == 2).sum())}
    payload = {"neuron_count": len(body_ids), "indices": torch.from_numpy(np.stack((target, source))).long(), "weights": torch.from_numpy(weight).float(), "body_ids": torch.from_numpy(body_ids).long(), "edge_groups": torch.from_numpy(groups).long(), "soma_positions": soma_positions(annotations, body_ids), "anatomy_regions": anatomy_regions(annotations, body_ids), "metadata": {"release": "MaleCNS v1.0", "minimum_weight": minimum_weight, "source": BASE, "edge_group_counts": counts, "soma_coordinate_space": "Male CNS EM, 8 nm voxels"}}
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    print(json.dumps({"neurons": len(body_ids), "edges": len(weight), "edge_groups": counts, "output": str(output)}, indent=2))

def augment_positions(raw: Path, graph_path: Path, output: Path | None = None):
    import pyarrow.feather as feather
    payload = torch.load(graph_path, map_location="cpu", weights_only=True)
    annotations = feather.read_table(raw / FILES["annotations"])
    positions = soma_positions(annotations, payload["body_ids"].numpy())
    payload["soma_positions"] = positions
    payload["anatomy_regions"] = anatomy_regions(annotations, payload["body_ids"].numpy())
    payload.setdefault("metadata", {})["soma_coordinate_space"] = "Male CNS EM, 8 nm voxels"
    destination = output or graph_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)
    valid = int(torch.isfinite(positions).all(1).sum())
    print(json.dumps({"neurons": int(payload["neuron_count"]), "soma_positions": valid, "output": str(destination)}, indent=2))

def main():
    p = argparse.ArgumentParser(); p.add_argument("command", choices=("download", "build", "all", "positions")); p.add_argument("--raw", type=Path, default=Path("data/raw/malecns-v1.0")); p.add_argument("--output", type=Path); p.add_argument("--graph", type=Path); p.add_argument("--minimum-weight", type=int, default=3); a = p.parse_args()
    if a.command in ("download", "all"): download(a.raw)
    if a.command in ("build", "all"): build(a.raw, a.output or Path("data/processed/malecns-v1.0.pt"), a.minimum_weight)
    if a.command == "positions":
        if a.graph is None: p.error("positions requires --graph")
        augment_positions(a.raw, a.graph, a.output)
