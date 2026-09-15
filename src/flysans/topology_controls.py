import argparse,json
from pathlib import Path
import torch

def make_control(graph,kind,seed):
    generator=torch.Generator().manual_seed(seed);result=dict(graph);indices=graph["indices"].clone();weights=graph["weights"].clone();groups=graph.get("edge_groups")
    if kind=="degree_rewire":
        # Directed configuration-model null: exact source/target degree multisets
        # before sparse coalescing. Duplicate count is reported, never hidden.
        indices[1]=indices[1,torch.randperm(indices.shape[1],generator=generator)]
    elif kind=="weight_shuffle":
        order=torch.randperm(weights.numel(),generator=generator);weights=weights[order]
        if groups is not None:groups=groups[order]
    elif kind=="random_sparse":
        n=int(graph["neuron_count"]);indices=torch.randint(n,indices.shape,generator=generator)
    else:raise ValueError(kind)
    pairs=indices[0].to(torch.int64)*int(graph["neuron_count"])+indices[1].to(torch.int64);unique=torch.unique(pairs).numel();result["indices"]=indices;result["weights"]=weights
    if groups is not None:result["edge_groups"]=groups
    result["control"]={"kind":kind,"seed":seed,"edges_before_coalesce":indices.shape[1],"duplicate_pairs":indices.shape[1]-unique}
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument("--input",required=True);p.add_argument("--kind",choices=("degree_rewire","weight_shuffle","random_sparse"),required=True);p.add_argument("--seed",type=int,required=True);p.add_argument("--output",required=True);a=p.parse_args();graph=torch.load(a.input,map_location="cpu",weights_only=True);control=make_control(graph,a.kind,a.seed);path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True);torch.save(control,path);print(json.dumps({"output":str(path),**control["control"]}))

if __name__=="__main__":main()
