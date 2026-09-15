import copy
import random
from dataclasses import dataclass

import torch

TAKE, PASS, ACTIONS = 36, 37, 38
RANKS = 9

def suit(card): return card // RANKS
def rank(card): return card % RANKS

def beats(card, attack, trump):
    return (suit(card) == suit(attack) and rank(card) > rank(attack)) or (suit(card) == trump and suit(attack) != trump)

@dataclass
class DurakState:
    hands: list
    deck: list
    trump: int
    attacker: int
    actor: int
    phase: str
    table: list
    discarded: list
    limit: int
    winner: int | None = None

class DurakGame:
    """Two-player 36-card podkidnoy durak with attack, defence, throw-in, draw and refill."""
    def __init__(self, seed=0): self.reset(seed)

    def reset(self, seed):
        rng=random.Random(seed); deck=list(range(36)); rng.shuffle(deck); trump=suit(deck[0])
        hands=[[deck.pop() for _ in range(6)],[deck.pop() for _ in range(6)]]
        trumps=[(rank(c),p) for p,h in enumerate(hands) for c in h if suit(c)==trump]
        attacker=min(trumps)[1] if trumps else 0
        self.s=DurakState(hands,deck,trump,attacker,attacker,"attack",[],[],min(6,len(hands[1-attacker])))
        return self.s

    def legal(self):
        s=self.s; legal=[]; hand=s.hands[s.actor]; ranks={rank(c) for pair in s.table for c in pair if c is not None}
        if s.phase in ("attack","throw"):
            legal=[c for c in hand if (not s.table or rank(c) in ranks) and len(s.table)<s.limit]
            if s.table: legal.append(PASS)
        else:
            attack=next(a for a,d in s.table if d is None)
            legal=[c for c in hand if beats(c,attack,s.trump)]+[TAKE]
        return legal

    def _refill(self, first):
        for player in (first,1-first):
            while len(self.s.hands[player])<6 and self.s.deck: self.s.hands[player].append(self.s.deck.pop())

    def _finish_check(self):
        if self.s.deck: return
        empty=[p for p in (0,1) if not self.s.hands[p]]
        if len(empty)==1: self.s.winner=empty[0]
        elif len(empty)==2: self.s.winner=-1

    def play(self, action):
        s=self.s
        if s.winner is not None: return "finished"
        if action not in self.legal(): raise ValueError(f"illegal action {action}; legal={self.legal()}")
        event="card"
        if s.phase=="defend":
            if action==TAKE:
                s.phase="throw"; s.actor=s.attacker; event="take"
            else:
                s.hands[s.actor].remove(action)
                for pair in s.table:
                    if pair[1] is None: pair[1]=action; break
                s.phase="attack"; s.actor=s.attacker
        elif action==PASS:
            defender=1-s.attacker
            if s.phase=="throw":
                s.hands[defender].extend(c for pair in s.table for c in pair if c is not None)
                next_attacker=s.attacker; event="taken_round"
            else:
                s.discarded.extend(c for pair in s.table for c in pair if c is not None)
                next_attacker=defender; event="defended_round"
            old_attacker=s.attacker; s.table=[]; self._refill(old_attacker); self._finish_check()
            if s.winner is None:
                s.attacker=next_attacker; s.actor=next_attacker; s.phase="attack"; s.limit=min(6,len(s.hands[1-next_attacker]))
        else:
            s.hands[s.actor].remove(action); s.table.append([action,None])
            if s.phase == "throw":
                s.actor=s.attacker
            else:
                s.phase="defend"; s.actor=1-s.attacker
        return event

    def _determinized_copy(self, observer, rng):
        """Sample hidden cards without revealing the other player's hand."""
        game=copy.deepcopy(self); s=game.s
        public={c for pair in s.table for c in pair if c is not None}|set(s.discarded)
        known=set(s.hands[observer])|public
        trump_card=s.deck[0] if s.deck else None
        if trump_card is not None: known.add(trump_card)
        unseen=list(set(range(36))-known); rng.shuffle(unseen)
        other=1-observer; other_count=len(s.hands[other]); deck_count=len(s.deck)
        s.hands[other]=unseen[:other_count]
        remaining=unseen[other_count:]
        if deck_count:
            s.deck=[trump_card]+remaining[:deck_count-1]
        else:s.deck=[]
        return game

    def _rollout_score(self, observer, rng, limit=256):
        for _ in range(limit):
            if self.s.winner is not None:break
            legal=self.legal(); cards=[a for a in legal if a<36]
            if cards:
                action=min(cards,key=lambda c:(suit(c)==self.s.trump,rank(c)))
            else:action=PASS if PASS in legal else TAKE
            self.play(action)
        if self.s.winner==observer:return 1.0
        if self.s.winner in (-1,None):
            other=1-observer
            return max(-.5,min(.5,(len(self.s.hands[other])-len(self.s.hands[observer]))/12))
        return -1.0

    def choose_search(self, rng, rollouts=16):
        """Root determinization search for two-player imperfect-information Durak."""
        observer=self.s.actor; legal=self.legal()
        if len(legal)==1:return legal[0]
        totals={a:0.0 for a in legal}; counts={a:0 for a in legal}
        # Every action receives samples; larger budgets improve stability/strength.
        for n in range(max(len(legal),rollouts)):
            action=legal[n%len(legal)]; game=self._determinized_copy(observer,rng)
            game.play(action)
            totals[action]+=game._rollout_score(observer,rng);counts[action]+=1
        return max(legal,key=lambda a:(totals[a]/counts[a],a in (PASS,TAKE),-a))

    def choose_opponent(self, policy, rng, search_rollouts=16):
        legal=self.legal()
        if policy=="random": return rng.choice(legal)
        if policy=="search":return self.choose_search(rng,search_rollouts)
        cards=[a for a in legal if a<36]
        if cards:
            if policy=="aggressive":
                if self.s.phase in ("attack","throw"):
                    return max(cards,key=lambda c:(suit(c)!=self.s.trump,rank(c)))
                return min(cards,key=lambda c:(suit(c)==self.s.trump,rank(c)))
            if policy=="trump_saver":
                non_trumps=[c for c in cards if suit(c)!=self.s.trump]
                if non_trumps:return min(non_trumps,key=rank)
                if self.s.phase=="defend" and self.s.deck and len(self.s.table)<=2 and TAKE in legal:return TAKE
                return min(cards,key=lambda c:(suit(c)==self.s.trump,rank(c)))
            if policy=="counter":
                unseen=set(range(36))-set(self.s.hands[self.s.actor])-set(self.s.discarded)-{c for pair in self.s.table for c in pair if c is not None}
                if self.s.phase in ("attack","throw"):
                    return min(cards,key=lambda c:(sum(beats(x,c,self.s.trump) for x in unseen),suit(c)==self.s.trump,rank(c)))
                return min(cards,key=lambda c:(suit(c)==self.s.trump,rank(c)))
            return min(cards,key=lambda c:(suit(c)==self.s.trump,rank(c)))
        return PASS if PASS in legal else TAKE

class DurakBatchEnv:
    observation_size=154
    action_size=ACTIONS
    def __init__(self,batch_size=256,opponent="heuristic",seed=0,device="cpu",opponent_model=None,search_rollouts=16,search_fraction=1.0):
        self.batch_size=batch_size; self.opponent=opponent; self.device=torch.device(device); self.seed=seed
        self.opponent_models=[] if opponent_model is None else (opponent_model if isinstance(opponent_model,list) else [opponent_model])
        self.search_rollouts=search_rollouts;self.search_fraction=search_fraction
        self.games=[]; self.rngs=[]; self.reset(seed)

    def reset(self,seed=None):
        if seed is not None:self.seed=seed
        self.games=[DurakGame(self.seed+i) for i in range(self.batch_size)]; self.rngs=[random.Random(self.seed+100000+i) for i in range(self.batch_size)]
        policies=("random","heuristic","aggressive","trump_saver","counter")
        if self.opponent=="league": self.policies=[f"neural:{(self.seed+i)%len(self.opponent_models)}" if i%2==0 and self.opponent_models else policies[(self.seed+i)%len(policies)] for i in range(self.batch_size)]
        elif self.opponent=="search":self.policies=["search" if (i+.5)/self.batch_size<=self.search_fraction else "counter" for i in range(self.batch_size)]
        else: self.policies=[policies[(self.seed+i)%len(policies)] if self.opponent=="mixed" else self.opponent for i in range(self.batch_size)]
        self.agent=[i%2 for i in range(self.batch_size)]; self.done=[False]*self.batch_size
        self.opponent_states=[model.initial_state(self.batch_size,self.device) for model in self.opponent_models]
        self._advance_opponents(); return self.observe()

    def _advance_opponents(self):
        events=[[] for _ in self.games]
        for guard in range(100):
            pending=[i for i,g in enumerate(self.games) if g.s.winner is None and g.s.actor!=self.agent[i]]
            if not pending: break
            neural=[i for i in pending if self.policies[i].startswith("neural:")]
            for model_index in range(len(self.opponent_models)):
                group=[i for i in neural if self.policies[i]==f"neural:{model_index}"]
                if not group:continue
                obs=self.observe(players=[1-self.agent[i] for i in range(self.batch_size)])
                idx=torch.tensor(group,device=self.device); legal=self.legal_mask(players=[1-self.agent[i] for i in range(self.batch_size)])[idx]
                with torch.no_grad():
                    logits,_,next_state=self.opponent_models[model_index](obs[idx],self.opponent_states[model_index][idx]); action=logits.masked_fill(~legal,-1e9).argmax(-1)
                self.opponent_states[model_index][idx]=next_state
                for i,a in zip(group,action.tolist()): events[i].append(self.games[i].play(a))
            for i in pending:
                if not self.policies[i].startswith("neural:"): events[i].append(self.games[i].play(self.games[i].choose_opponent(self.policies[i],self.rngs[i],self.search_rollouts)))
        else: raise RuntimeError("opponent loop did not yield")
        for i,g in enumerate(self.games): self.done[i]=g.s.winner is not None
        return events

    def legal_mask(self,players=None):
        mask=torch.zeros(self.batch_size,ACTIONS,dtype=torch.bool,device=self.device)
        for i,g in enumerate(self.games):
            if self.done[i]: mask[i,PASS]=True
            else: mask[i,g.legal()]=True
        return mask

    def observe(self,players=None):
        obs=torch.zeros(self.batch_size,self.observation_size,dtype=torch.float32)
        for i,g in enumerate(self.games):
            s=g.s; me=self.agent[i] if players is None else players[i]
            obs[i,s.hands[me]]=1
            for a,d in s.table:
                obs[i,36+a]=1
                if d is not None: obs[i,72+d]=1
            obs[i,108+s.trump]=1
            obs[i,112+(0 if s.phase=="attack" else 1 if s.phase=="defend" else 2)]=1
            obs[i,115]=len(s.deck)/24; obs[i,116]=len(s.hands[1-me])/36; obs[i,117]=float(s.attacker==me)
            for card in s.discarded: obs[i,118+card]=1
        return obs.to(self.device)

    def step(self,actions):
        rewards=torch.zeros(self.batch_size,device=self.device)
        snapshots=[]; own_events=[]
        for i,(g,a) in enumerate(zip(self.games,actions.tolist())):
            me=self.agent[i]
            snapshots.append((len(g.s.hands[1-me])-len(g.s.hands[me]))/36)
            own_events.append(None if self.done[i] else g.play(a))
        all_opponent_events=self._advance_opponents()
        for i,g in enumerate(self.games):
            if own_events[i] is None: continue
            me=self.agent[i]; before=snapshots[i]; own_event=own_events[i]; opponent_events=all_opponent_events[i]
            after=(len(g.s.hands[1-me])-len(g.s.hands[me]))/36
            rewards[i]+=0.02*(after-before)
            if own_event=="take": rewards[i]-=0.03
            if "take" in opponent_events: rewards[i]+=0.03
            if own_event=="defended_round": rewards[i]+=0.01
            if "defended_round" in opponent_events: rewards[i]-=0.01
            if g.s.winner is not None: rewards[i]=1.0 if g.s.winner==self.agent[i] else 0.0 if g.s.winner==-1 else -1.0
        return self.observe(),rewards,torch.tensor(self.done,device=self.device),{"legal_mask":self.legal_mask()}
