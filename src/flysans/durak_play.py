import argparse,random
import torch
from .connectome import load_graph
from .durak import DurakGame,PASS,TAKE
from .model import DeepFlyPolicy,FlyPolicy

RANK_NAMES=("6","7","8","9","10","J","Q","K","A")
SUIT_NAMES=("♣","♦","♥","♠")

def card_name(card):return RANK_NAMES[card%9]+SUIT_NAMES[card//9]

def observation(game,player,device):
    s=game.s;obs=torch.zeros(1,154,device=device);obs[0,s.hands[player]]=1
    for attack,defence in s.table:
        obs[0,36+attack]=1
        if defence is not None:obs[0,72+defence]=1
    obs[0,108+s.trump]=1;obs[0,112+(0 if s.phase=="attack" else 1 if s.phase=="defend" else 2)]=1
    obs[0,115]=len(s.deck)/24;obs[0,116]=len(s.hands[1-player])/36;obs[0,117]=float(s.attacker==player)
    for card in s.discarded:obs[0,118+card]=1
    return obs

def action_name(action):
    if action==TAKE:return "БЕРУ"
    if action==PASS:return "ПАС"
    return card_name(action)

def print_position(game,human):
    s=game.s;table=" ".join(f"{card_name(a)}/{card_name(d) if d is not None else '—'}" for a,d in s.table) or "пусто"
    print(f"\nКозырь: {SUIT_NAMES[s.trump]} | Колода: {len(s.deck)} | У соперника: {len(s.hands[1-human])}")
    print(f"Стол: {table}");print("Твоя рука:"," ".join(card_name(c) for c in sorted(s.hands[human])))

def main():
    p=argparse.ArgumentParser();p.add_argument("--graph",required=True);p.add_argument("--checkpoint",required=True);p.add_argument("--device",default="cuda");p.add_argument("--seed",type=int);p.add_argument("--human-seat",choices=(0,1),type=int);a=p.parse_args();device=torch.device(a.device);seed=a.seed if a.seed is not None else random.randrange(2**31);human=a.human_seat if a.human_seat is not None else random.randrange(2);ai=1-human
    checkpoint=torch.load(a.checkpoint,map_location=device,weights_only=True);graph=load_graph(a.graph,device);model_class=DeepFlyPolicy if checkpoint.get("architecture")=="deep" else FlyPolicy;model=model_class(graph,154,38,plastic_edges=checkpoint.get("plastic_edges",False)).to(device).eval();model.load_state_dict(checkpoint["model"]);state=model.initial_state(1,device);game=DurakGame(seed)
    print(f"Раздача seed={seed}. Ты игрок {human}; муха игрок {ai}. Введи номер варианта, q — выйти.")
    while game.s.winner is None:
        if game.s.actor==human:
            print_position(game,human);legal=game.legal()
            for number,action in enumerate(legal,1):print(f"  {number}: {action_name(action)}")
            while True:
                raw=input("Твой ход> ").strip().lower()
                if raw in ("q","quit","exit"):return
                try:index=int(raw)-1
                except ValueError:index=-1
                if 0<=index<len(legal):break
                print("Выбери номер из списка.")
            game.play(legal[index])
        else:
            obs=observation(game,ai,device);legal=game.legal();mask=torch.zeros(1,38,dtype=torch.bool,device=device);mask[0,legal]=True
            with torch.no_grad():logits,_,state=model(obs,state);action=int(logits.masked_fill(~mask,-1e9).argmax(-1).item())
            print_position(game,human);print("Муха ходит:",action_name(action));game.play(action)
    print_position(game,human)
    if game.s.winner==-1:print("Ничья: оба вышли одновременно.")
    elif game.s.winner==human:print("Ты победил — муха осталась дураком.")
    else:print("Муха победила. Сегодня дурак — человек.")

if __name__=="__main__":main()
