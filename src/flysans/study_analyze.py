import argparse,json,math
from collections import defaultdict
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument("--root",required=True);p.add_argument("--output",default="study-summary.md");a=p.parse_args();root=Path(a.root);rows=defaultdict(lambda:defaultdict(list));metadata={}
    for log in (root/"logs").glob("*.jsonl"):
        condition=log.stem.rsplit("-s",1)[0]
        for line in log.read_text(encoding="utf-8").splitlines():
            try:item=json.loads(line)
            except json.JSONDecodeError:continue
            if "trainable_parameters" in item:metadata[condition]=item
            if "opponent" in item and "win_rate" in item:rows[condition][item["opponent"]].append(item["win_rate"])
    lines=["# Study summary","", "| Condition | Parameters | Opponent | Seeds | Mean win rate | 95% normal CI |","|---|---:|---|---:|---:|---:|"]
    for condition in sorted(rows):
        for opponent,values in sorted(rows[condition].items()):
            n=len(values);mean=sum(values)/n;sd=math.sqrt(sum((x-mean)**2 for x in values)/(n-1)) if n>1 else float("nan");ci=1.96*sd/math.sqrt(n) if n>1 else float("nan");lines.append(f"| {condition} | {metadata.get(condition,{}).get('trainable_parameters','?')} | {opponent} | {n} | {mean:.4f} | ±{ci:.4f} |")
    output=Path(a.output);output.write_text("\n".join(lines)+"\n",encoding="utf-8");print(json.dumps({"output":str(output),"conditions":len(rows)}))

if __name__=="__main__":main()
