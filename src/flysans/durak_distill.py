import argparse,json
from pathlib import Path
import torch
from torch.nn import functional as F
from .connectome import load_graph
from .durak import DurakBatchEnv
from .model import DeepFlyPolicy,FlyPolicy,MLPPolicy

def main():
    p=argparse.ArgumentParser();p.add_argument("--graph",required=True);p.add_argument("--teacher",required=True);p.add_argument("--updates",type=int,default=50);p.add_argument("--batch-size",type=int,default=128);p.add_argument("--decisions",type=int,default=128);p.add_argument("--learning-rate",type=float,default=3e-4);p.add_argument("--device",default="cuda");p.add_argument("--output",default="runs/durak-student.pt");a=p.parse_args();device=torch.device(a.device)
    graph=load_graph(a.graph,device);checkpoint=torch.load(a.teacher,map_location=device,weights_only=True);teacher_class=DeepFlyPolicy if checkpoint.get("architecture")=="deep" else FlyPolicy;teacher=teacher_class(graph,DurakBatchEnv.observation_size,DurakBatchEnv.action_size,plastic_edges=checkpoint.get("plastic_edges",False)).to(device).eval();teacher.load_state_dict(checkpoint["model"])
    for parameter in teacher.parameters():parameter.requires_grad_(False)
    student=MLPPolicy(DurakBatchEnv.observation_size,DurakBatchEnv.action_size).to(device);opt=torch.optim.Adam(student.parameters(),lr=a.learning_rate)
    for update in range(a.updates):
        env=DurakBatchEnv(a.batch_size,"mixed",700000+update,device);obs=env.reset(700000+update);state=teacher.initial_state(a.batch_size,device);active=torch.ones(a.batch_size,dtype=torch.bool,device=device);losses=[];agreements=[]
        for step in range(a.decisions):
            with torch.no_grad():teacher_logits,_,state=teacher(obs,state)
            legal=env.legal_mask();masked_teacher=teacher_logits.masked_fill(~legal,-1e9);student_logits,_,_=student(obs,student.initial_state(a.batch_size,device));masked_student=student_logits.masked_fill(~legal,-1e9);valid=active
            loss=F.kl_div(F.log_softmax(masked_student[valid],-1),F.softmax(masked_teacher[valid],-1),reduction="batchmean");opt.zero_grad(set_to_none=True);loss.backward();opt.step();losses.append(float(loss.detach()));agreements.append(float((masked_student[valid].argmax(-1)==masked_teacher[valid].argmax(-1)).float().mean()))
            action=masked_teacher.argmax(-1);obs,_,done,_=env.step(action);active=active&~done;state=state*active[:,None]
            if not bool(active.any()):break
        print(json.dumps({"distill_update":update+1,"loss":sum(losses)/len(losses),"action_agreement":sum(agreements)/len(agreements)}),flush=True)
    path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True);torch.save({"model":student.state_dict(),"teacher":a.teacher},path);print(json.dumps({"student":str(path)}))

if __name__=="__main__":main()
