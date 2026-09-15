import argparse,json,time
from dataclasses import asdict,dataclass
from pathlib import Path
import torch
from torch.distributions import Categorical
from torch.nn import functional as F
from .connectome import load_graph
from .durak import DurakBatchEnv
from .model import DeepFlyPolicy,FlyPolicy,MLPPolicy,RecurrentPolicy

@dataclass(frozen=True)
class Config:
    updates:int=1000; batch_size:int=256; decisions:int=128; learning_rate:float=3e-5; entropy_weight:float=0.01; seed:int=1; eval_games:int=2048; gamma:float=.995; gae_lambda:float=.95

def rollout(model,env,seed,decisions=128,greedy=False,teacher_fraction=0.0):
    obs=env.reset(seed); state=model.initial_state(env.batch_size,env.device); active=torch.ones(env.batch_size,dtype=torch.bool,device=env.device)
    logs=[]; values=[]; rewards=[]; entropy=[]; normalized_entropy=[]; masks=[]; dones=[]; wins=torch.zeros(env.batch_size,device=env.device);teacher_losses=[];teacher_matches=[]
    for step in range(decisions):
        if step and step%8==0: state=state.detach()
        logits,value,state=model(obs,state); mask=env.legal_mask(); logits=logits.masked_fill(~mask,-1e9)
        if not greedy and env.opponent=="search" and teacher_fraction>0:
            active_list=active.tolist()
            stride=max(1,round(1/teacher_fraction))
            teacher_indices=[i for i,g in enumerate(env.games) if active_list[i] and g.s.winner is None and g.s.actor==env.agent[i] and (i+step)%stride==0]
            for i in teacher_indices:
                target=env.games[i].choose_search(env.rngs[i],env.search_rollouts)
                teacher_losses.append(-F.log_softmax(logits[i],-1)[target]);teacher_matches.append((logits[i].argmax()==target).float())
        dist=Categorical(logits=logits); action=logits.argmax(-1) if greedy else dist.sample()
        obs,reward,done,_=env.step(action); reward*=active; wins=torch.where(done&active,(reward>0).float(),wins)
        action_count=mask.sum(-1).float(); normalizer=action_count.log().clamp_min(1)
        normalized_entropy.append(torch.where(action_count>1,dist.entropy()/normalizer,torch.zeros_like(action_count)))
        masks.append(active.float()); dones.append(done.float()); logs.append(dist.log_prob(action)); values.append(value); rewards.append(reward); entropy.append(dist.entropy())
        active = active & ~done; state = state * active[:,None]
        if not bool(active.any()): break
    teacher_loss=torch.stack(teacher_losses).mean() if teacher_losses else logs[0].sum()*0
    teacher_agreement=float(torch.stack(teacher_matches).mean()) if teacher_matches else 0.0
    return torch.stack(logs),torch.stack(values),torch.stack(rewards),torch.stack(entropy),torch.stack(normalized_entropy),torch.stack(masks),torch.stack(dones),teacher_loss,teacher_agreement,wins,1-active.float()

def gae_targets(values,rewards,dones,masks,gamma=.995,lam=.95):
    advantages=torch.zeros_like(values); running=torch.zeros(values.shape[1],device=values.device)
    for t in range(values.shape[0]-1,-1,-1):
        next_value=values[t+1].detach() if t+1<values.shape[0] else torch.zeros_like(values[t])
        continuation=1-dones[t]
        delta=rewards[t]+gamma*next_value*continuation-values[t]
        running=(delta+gamma*lam*continuation*running)*masks[t]
        advantages[t]=running
    return advantages,advantages+values

def save(path,model,opt,cfg,update,model_type,architecture="base",plastic_edges=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);torch.save({"model":model.state_dict(),"optimizer":opt.state_dict(),"config":asdict(cfg),"update":update,"model_type":model_type,"architecture":architecture,"plastic_edges":plastic_edges},path)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--graph");p.add_argument("--model",choices=("connectome","mlp","gru","lstm","rnn"),default="connectome")
    p.add_argument("--architecture",choices=("base","deep"),default="base");p.add_argument("--plastic-edges",action="store_true")
    p.add_argument("--edge-learning-rate",type=float,default=1e-6);p.add_argument("--plasticity-weight",type=float,default=1e-4);p.add_argument("--hidden-size",type=int,default=1024)
    p.add_argument("--opponent",choices=("random","heuristic","aggressive","trump_saver","counter","mixed","league","search"),default="mixed")
    p.add_argument("--opponent-checkpoint",action="append",default=[]);p.add_argument("--search-rollouts",type=int,default=16);p.add_argument("--search-fraction",type=float,default=.25)
    p.add_argument("--teacher-weight",type=float,default=0.0);p.add_argument("--teacher-fraction",type=float,default=.125)
    p.add_argument("--updates",type=int,default=1000);p.add_argument("--batch-size",type=int,default=256);p.add_argument("--decisions",type=int,default=128)
    p.add_argument("--learning-rate",type=float,default=3e-5);p.add_argument("--entropy-weight",type=float,default=0.01);p.add_argument("--eval-games",type=int,default=2048)
    p.add_argument("--seed",type=int,default=1);p.add_argument("--resume");p.add_argument("--device",default="cuda");p.add_argument("--output",default="runs/durak.pt")
    a=p.parse_args();device=torch.device(a.device);cfg=Config(a.updates,a.batch_size,a.decisions,a.learning_rate,a.entropy_weight,a.seed,a.eval_games)
    torch.manual_seed(a.seed)
    if device.type=="cuda":torch.cuda.manual_seed_all(a.seed)
    if a.model=="connectome" and not a.graph:p.error("--graph is required for --model connectome")
    graph=load_graph(a.graph,device) if a.model=="connectome" else None
    if a.model=="connectome": model=(DeepFlyPolicy if a.architecture=="deep" else FlyPolicy)(graph,DurakBatchEnv.observation_size,DurakBatchEnv.action_size,plastic_edges=a.plastic_edges)
    elif a.model=="mlp": model=MLPPolicy(DurakBatchEnv.observation_size,DurakBatchEnv.action_size,a.hidden_size)
    else: model=RecurrentPolicy(DurakBatchEnv.observation_size,DurakBatchEnv.action_size,a.hidden_size,a.model)
    model=model.to(device);plastic_names={"source_gain_logits","target_gain_logits"};edge_parameters=[p for n,p in model.named_parameters() if n in plastic_names];other_parameters=[p for n,p in model.named_parameters() if n not in plastic_names];groups=[{"params":other_parameters,"lr":cfg.learning_rate}]
    if edge_parameters:groups.append({"params":edge_parameters,"lr":a.edge_learning_rate})
    opt=torch.optim.Adam(groups);start=0
    if a.resume:
        s=torch.load(a.resume,map_location=device,weights_only=True);missing,unexpected=model.load_state_dict(s["model"],strict=False);start=s["update"]+1
        if not missing and not unexpected:
            try:opt.load_state_dict(s["optimizer"])
            except ValueError:pass
        print(json.dumps({"resume":a.resume,"new_parameters":missing,"ignored_parameters":unexpected}),flush=True)
        for index,group in enumerate(opt.param_groups): group["lr"]=a.edge_learning_rate if index and edge_parameters else cfg.learning_rate
    opponent_model=[]
    if a.opponent=="league":
        if not a.opponent_checkpoint:p.error("--opponent-checkpoint is required for league")
        for checkpoint in a.opponent_checkpoint:
            snapshot=MLPPolicy(DurakBatchEnv.observation_size,DurakBatchEnv.action_size).to(device).eval()
            snapshot.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=True)["model"])
            for parameter in snapshot.parameters():parameter.requires_grad_(False)
            opponent_model.append(snapshot)
    parameter_count=sum(p.numel() for p in model.parameters() if p.requires_grad);print(json.dumps({"model":a.model,"architecture":a.architecture,"hidden_size":a.hidden_size,"seed":a.seed,"trainable_parameters":parameter_count}),flush=True)
    if device.type=="cuda":torch.cuda.reset_peak_memory_stats(device)
    best=-1.0; recent=[]
    for update in range(start,start+cfg.updates):
        update_started=time.perf_counter()
        if a.plastic_edges:model.set_plasticity_window(update,.025)
        env=DurakBatchEnv(cfg.batch_size,a.opponent,cfg.seed+update,device,opponent_model,a.search_rollouts,a.search_fraction);lp,v,reward,ent,nent,mask,dones,teacher_loss,teacher_agreement,wins,finished=rollout(model,env,cfg.seed+update,cfg.decisions,teacher_fraction=a.teacher_fraction if a.teacher_weight>0 else 0);adv,ret=gae_targets(v,reward,dones,mask,cfg.gamma,cfg.gae_lambda);valid=mask.bool();norm=torch.zeros_like(adv);norm[valid]=(adv[valid]-adv[valid].mean())/(adv[valid].std(unbiased=False)+1e-6);pl=-(lp[valid]*norm[valid].detach()).mean();vl=F.smooth_l1_loss(v[valid],ret[valid].detach());plasticity=model.plasticity_penalty() if hasattr(model,"plasticity_penalty") else pl.new_zeros(());loss=pl+.5*vl-cfg.entropy_weight*ent[valid].mean()+a.plasticity_weight*plasticity+a.teacher_weight*teacher_loss;opt.zero_grad(set_to_none=True);loss.backward();gn=torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();search_indices=[i for i,p in enumerate(env.policies) if p=="search"];search_win=float(wins[search_indices].mean()) if search_indices else 0.0;selection_score=search_win if a.opponent=="search" else float(wins.mean());recent.append(selection_score);recent=recent[-20:];rolling=sum(recent)/len(recent)
        peak=round(torch.cuda.max_memory_allocated(device)/2**30,3) if device.type=="cuda" else 0
        edge_stats={}
        if edge_parameters:
            source_gain=0.5+torch.sigmoid(model.source_gain_logits.detach());target_gain=0.5+torch.sigmoid(model.target_gain_logits.detach());edge_stats={"source_gain_mean":float(source_gain.mean()),"source_gain_std":float(source_gain.std()),"target_gain_mean":float(target_gain.mean()),"target_gain_std":float(target_gain.std()),"plasticity_penalty":float(plasticity.detach()),"active_edges":model.edge_window_count,"edge_coverage_updates":1,"plasticity_mode":"factorized"}
        print(json.dumps({"update":update+1,"loss":float(loss.detach()),"policy_loss":float(pl.detach()),"value_loss":float(vl.detach()),"teacher_loss":float(teacher_loss.detach()),"teacher_agreement":teacher_agreement,"teacher_weight":a.teacher_weight,"teacher_fraction":a.teacher_fraction if a.teacher_weight>0 else 0,"entropy":float(ent[valid].mean().detach()),"normalized_entropy":float(nent[valid].mean().detach()),"grad_norm":float(gn),"win_rate":float(wins.mean()),"search_win_rate":search_win if a.opponent=="search" else None,"rolling_win_rate_20":rolling,"search_rollouts":a.search_rollouts if a.opponent=="search" else 0,"search_fraction":a.search_fraction if a.opponent=="search" else 0,"update_seconds":round(time.perf_counter()-update_started,2),"finished_rate":float(finished.mean()),"peak_gpu_gib":peak,**edge_stats}),flush=True)
        if (update+1)%10==0:save(a.output,model,opt,cfg,update,a.model,a.architecture,a.plastic_edges)
        if len(recent)==20 and rolling>best:
            best=rolling;save(str(Path(a.output).with_suffix(".best.pt")),model,opt,cfg,update,a.model,a.architecture,a.plastic_edges)
    save(a.output,model,opt,cfg,start+cfg.updates-1,a.model,a.architecture,a.plastic_edges)
    for opponent in ("random","heuristic","aggressive","trump_saver","counter"):
        wins=[]
        for offset in range(0,cfg.eval_games,cfg.batch_size):
            size=min(cfg.batch_size,cfg.eval_games-offset);env=DurakBatchEnv(size,opponent,cfg.seed+1000000+offset,device)
            with torch.no_grad(): *_,w,finished=rollout(model,env,cfg.seed+1000000+offset,cfg.decisions,True)
            wins.append(w);print(json.dumps({"evaluation_opponent":opponent,"completed":offset+size,"total":cfg.eval_games}),flush=True)
        all_wins=torch.cat(wins);even=all_wins[::2];odd=all_wins[1::2]
        print(json.dumps({"opponent":opponent,"win_rate":float(all_wins.mean()),"seat_0":float(even.mean()),"seat_1":float(odd.mean())}),flush=True)

if __name__=="__main__":main()
