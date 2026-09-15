import argparse,json,subprocess,sys,time
from pathlib import Path

CONDITIONS=(
    ("malecns","connectome",None),
    ("degree-rewire","connectome","degree_rewire"),
    ("weight-shuffle","connectome","weight_shuffle"),
    ("random-sparse","connectome","random_sparse"),
    ("gru","gru",2816),("lstm","lstm",2528),("rnn","rnn",3968),("mlp","mlp",4096),
)

def run(command,log):
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open("w",encoding="utf-8") as stream:return subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT).returncode

def main():
    p=argparse.ArgumentParser();p.add_argument("--graph",required=True);p.add_argument("--output-root",default="runs/paper-pilot");p.add_argument("--seeds",default="1,2,3,4,5");p.add_argument("--updates",type=int,default=100);p.add_argument("--batch-size",type=int,default=512);p.add_argument("--decisions",type=int,default=128);p.add_argument("--eval-games",type=int,default=2048);p.add_argument("--opponent-checkpoint",action="append",default=[]);p.add_argument("--device",default="cuda");a=p.parse_args();root=Path(a.output_root);root.mkdir(parents=True,exist_ok=True);seeds=[int(x) for x in a.seeds.split(",")];records=[]
    for seed in seeds:
        for name,model,extra in CONDITIONS:
            graph=a.graph
            if isinstance(extra,str):
                graph=str(root/"graphs"/f"{name}-s{seed}.pt");path=Path(graph)
                if not path.exists():
                    path.parent.mkdir(parents=True,exist_ok=True);subprocess.run([sys.executable,"-m","flysans.topology_controls","--input",a.graph,"--kind",extra,"--seed",str(seed),"--output",graph],check=True)
            out=root/"checkpoints"/f"{name}-s{seed}.pt";log=root/"logs"/f"{name}-s{seed}.jsonl"
            command=[sys.executable,"-m","flysans.durak_train","--device",a.device,"--model",model,"--opponent","league","--updates",str(a.updates),"--batch-size",str(a.batch_size),"--decisions",str(a.decisions),"--eval-games",str(a.eval_games),"--seed",str(seed),"--output",str(out)]
            if model=="connectome":command += ["--graph",graph]
            else:command += ["--hidden-size",str(extra)]
            for checkpoint in a.opponent_checkpoint:command += ["--opponent-checkpoint",checkpoint]
            started=time.time();code=run(command,log);record={"condition":name,"seed":seed,"returncode":code,"seconds":time.time()-started,"checkpoint":str(out),"log":str(log)};records.append(record);(root/"manifest.json").write_text(json.dumps(records,indent=2),encoding="utf-8");print(json.dumps(record),flush=True)
            if code:raise SystemExit(f"failed: {name} seed {seed}; see {log}")

if __name__=="__main__":main()
