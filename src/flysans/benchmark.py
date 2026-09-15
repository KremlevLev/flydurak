import argparse, json, time
import torch
from .connectome import load_graph, make_synthetic_graph
from .env import EnvConfig, SansFightEnv
from .model import FlyPolicy

def main():
    p=argparse.ArgumentParser(); p.add_argument("--graph"); p.add_argument("--batch-size",type=int,default=64); p.add_argument("--steps",type=int,default=200); p.add_argument("--device",default="cuda"); a=p.parse_args()
    device=torch.device(a.device); graph=load_graph(a.graph,device) if a.graph else make_synthetic_graph().to(device)
    env=SansFightEnv(EnvConfig(a.batch_size,curriculum_level=4),device); model=FlyPolicy(graph,env.observation_size,env.action_size).to(device).eval(); obs,state=env.reset(1),model.initial_state(a.batch_size,device)
    if device.type=="cuda": torch.cuda.synchronize()
    start=time.perf_counter()
    with torch.no_grad():
        for _ in range(a.steps):
            logits,_,state=model(obs,state); obs,_,done,_=env.step(logits.argmax(-1)); state*= (~done)[:,None]
    if device.type=="cuda": torch.cuda.synchronize()
    seconds=time.perf_counter()-start
    print(json.dumps({"steps_per_second":a.steps*a.batch_size/seconds,"seconds":seconds,"neurons":graph.neuron_count,"edges":graph.weights.numel(),"batch_size":a.batch_size,"device":str(device)},indent=2))
