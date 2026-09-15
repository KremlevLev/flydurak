import argparse, json, random
from dataclasses import asdict, dataclass
from pathlib import Path
import torch
from torch.distributions import Categorical
from torch.nn import functional as F
from .connectome import load_graph, make_synthetic_graph
from .env import EnvConfig, SansFightEnv
from .model import FlyPolicy, MLPPolicy

@dataclass(frozen=True)
class TrainConfig:
    updates: int = 100
    batch_size: int = 64
    horizon: int = 180
    neurons: int = 256
    edges_per_neuron: int = 8
    learning_rate: float = 3e-4
    entropy_weight: float = 0.01
    seed: int = 1
    curriculum_every: int = 200
    eval_episodes: int = 512

def choose_device(requested): return torch.device("cuda" if requested == "auto" and torch.cuda.is_available() else "cpu" if requested == "auto" else requested)

def rollout(model, env, seed, mode="sample", bptt_steps=8):
    obs, state = env.reset(seed), model.initial_state(env.config.batch_size, env.device)
    logs, values, rewards, entropies, actions = [], [], [], [], []
    active = torch.ones(env.config.batch_size, dtype=torch.bool, device=env.device)
    episode_return = torch.zeros(env.config.batch_size, device=env.device)
    final = None
    for step in range(env.config.horizon):
        if step and step % bptt_steps == 0:
            state = state.detach()
        logits, value, state = model(obs, state); distribution = Categorical(logits=logits)
        if mode == "sample": action = distribution.sample()
        elif mode == "greedy": action = logits.argmax(-1)
        elif mode == "stationary": action = torch.full((env.config.batch_size,), 4, dtype=torch.long, device=env.device)
        elif mode == "random": action = torch.randint(env.action_size, (env.config.batch_size,), device=env.device)
        elif mode == "oracle": action = env.oracle_action()
        obs, reward, done, final = env.step(action)
        reward = reward * active.float(); episode_return += reward
        logs.append(distribution.log_prob(action)); values.append(value); rewards.append(reward); entropies.append(distribution.entropy()); actions.append(action)
        active = active & ~done
        state = state * active[:, None]
    returns = torch.stack(rewards).flip(0).cumsum(0).flip(0)
    return torch.stack(logs), torch.stack(values), returns, torch.stack(entropies), final["survived"].float(), episode_return, final["attack"], torch.stack(actions)

def evaluate(model, config, device, level, episodes=None):
    episodes = episodes or config.eval_episodes; batch = min(config.batch_size, episodes); results = {}
    for mode in ("greedy", "sample", "oracle", "random", "stationary"):
        survived, returns, attacks = [], [], []
        for offset in range(0, episodes, batch):
            size = min(batch, episodes-offset); env=SansFightEnv(EnvConfig(size,config.horizon,curriculum_level=level),device)
            with torch.no_grad(): _,_,_,_,s,r,a,_=rollout(model,env,config.seed+1_000_000+offset,mode)
            survived.append(s); returns.append(r); attacks.append(a)
            print(json.dumps({"evaluation_mode":mode,"completed":offset+size,"total":episodes}),flush=True)
        s,r,a=torch.cat(survived),torch.cat(returns),torch.cat(attacks)
        results[mode]={"survival":float(s.mean()),"mean_return":float(r.mean()),"by_attack":{SansFightEnv.attack_names[i]:float(s[a==i].mean()) for i in range(level+1) if bool((a==i).any())}}
    return results

def train(config, device, graph_path=None, resume=None, output=None, model_type="connectome", forced_level=None, train_attack=None):
    torch.manual_seed(config.seed); random.seed(config.seed)
    graph=load_graph(graph_path,device) if graph_path else make_synthetic_graph(config.neurons,config.edges_per_neuron,config.seed).to(device)
    model=(FlyPolicy(graph,SansFightEnv.observation_size,SansFightEnv.action_size) if model_type=="connectome" else MLPPolicy(SansFightEnv.observation_size,SansFightEnv.action_size)).to(device); optimizer=torch.optim.Adam(model.parameters(),lr=config.learning_rate); start=0
    if resume:
        saved=torch.load(resume,map_location=device,weights_only=True); model.load_state_dict(saved["model"]); optimizer.load_state_dict(saved["optimizer"]); start=int(saved["update"])+1
    last_loss=0.0
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
    for update in range(start,start+config.updates):
        level=forced_level if forced_level is not None else min(update//config.curriculum_every,len(SansFightEnv.attack_names)-1)
        attack_id = None if train_attack is None else SansFightEnv.attack_names.index(train_attack)
        env=SansFightEnv(EnvConfig(config.batch_size,config.horizon,curriculum_level=level,forced_attack=attack_id),device)
        logp,value,returns,entropy,train_survival,_,_,actions=rollout(model,env,config.seed+update,"sample")
        advantage=returns-value
        normalized=(advantage-advantage.mean())/(advantage.std(unbiased=False)+1e-6)
        policy_loss=-(logp*normalized.detach()).mean()
        value_loss=F.smooth_l1_loss(value,returns)
        entropy_weight=config.entropy_weight
        loss=policy_loss+0.5*value_loss-entropy_weight*entropy.mean()
        if not torch.isfinite(loss): raise RuntimeError(f"non-finite loss at update {update}")
        optimizer.zero_grad(set_to_none=True); loss.backward(); grad_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step(); last_loss=float(loss.detach())
        peak_gib = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0.0
        histogram=torch.bincount(actions.flatten(),minlength=SansFightEnv.action_size).float(); histogram/=histogram.sum()
        gains=getattr(model,"synaptic_gain",torch.empty(0,device=device)).detach().tolist()
        print(json.dumps({"update":update+1,"level":level,"loss":last_loss,"policy_loss":float(policy_loss.detach()),"value_loss":float(value_loss.detach()),"entropy":float(entropy.mean().detach()),"entropy_weight":entropy_weight,"grad_norm":float(grad_norm),"train_survival":float(train_survival.mean()),"action_fractions":[round(float(x),4) for x in histogram],"raw_gains":gains,"peak_gpu_gib":round(peak_gib,3)}),flush=True)
        if output and ((update+1)%10==0 or update==start+config.updates-1): save_checkpoint(output,model,optimizer,config,update,graph_path)
    level=forced_level if forced_level is not None else min((start+config.updates-1)//config.curriculum_every,len(SansFightEnv.attack_names)-1)
    metrics={"device":str(device),"loss":last_loss,"update":start+config.updates-1,"curriculum_level":level,"neurons":graph.neuron_count,"edges":graph.weights.numel(),"evaluation":evaluate(model,config,device,level)}
    return model,metrics,optimizer

def save_checkpoint(path,model,optimizer,config,update,graph_path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); torch.save({"model":model.state_dict(),"optimizer":optimizer.state_dict(),"config":asdict(config),"update":update,"graph_path":graph_path},path)

def main():
    p=argparse.ArgumentParser(); p.add_argument("--updates",type=int,default=100); p.add_argument("--batch-size",type=int,default=64); p.add_argument("--horizon",type=int,default=180); p.add_argument("--neurons",type=int,default=256); p.add_argument("--edges-per-neuron",type=int,default=8); p.add_argument("--learning-rate",type=float,default=1e-4); p.add_argument("--model",choices=("connectome","mlp"),default="connectome"); p.add_argument("--device",default="auto"); p.add_argument("--graph"); p.add_argument("--resume"); p.add_argument("--curriculum-every",type=int,default=200); p.add_argument("--curriculum-level",type=int,choices=range(5)); p.add_argument("--train-attack",choices=SansFightEnv.attack_names); p.add_argument("--eval-episodes",type=int,default=512); p.add_argument("--output",default="runs/poc.pt"); a=p.parse_args()
    config=TrainConfig(a.updates,a.batch_size,a.horizon,a.neurons,a.edges_per_neuron,learning_rate=a.learning_rate,curriculum_every=a.curriculum_every,eval_episodes=a.eval_episodes); device=choose_device(a.device)
    model,metrics,optimizer=train(config,device,a.graph,a.resume,a.output,a.model,a.curriculum_level,a.train_attack); save_checkpoint(a.output,model,optimizer,config,metrics["update"],a.graph); print(json.dumps({**metrics,"checkpoint":a.output},indent=2))

if __name__ == "__main__": main()
