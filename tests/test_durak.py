import torch
from flysans.durak import DurakBatchEnv,DurakGame,beats

def test_lower_card_never_beats_higher_card_of_same_suit_even_when_trump():
    assert not beats(27, 28, 3)  # 6♠ cannot beat 7♠ when spades are trump

def test_every_state_has_a_legal_action():
    env=DurakBatchEnv(32,"heuristic",3)
    for _ in range(200):
        mask=env.legal_mask();assert bool(mask.any(1).all());action=mask.float().argmax(1);_,_,done,_=env.step(action)
        if bool(done.all()):break

def test_deal_preserves_36_unique_cards():
    g=DurakGame(9);cards=g.s.deck+g.s.hands[0]+g.s.hands[1]
    assert len(cards)==36 and len(set(cards))==36

def test_search_returns_legal_action_without_changing_game():
    import random
    g=DurakGame(12);before=(list(g.s.deck),[list(h) for h in g.s.hands],g.s.phase)
    legal=g.legal();action=g.choose_search(random.Random(7),8)
    assert action in legal
    assert before==(g.s.deck,g.s.hands,g.s.phase)

def test_search_batch_opponent_runs():
    env=DurakBatchEnv(4,"search",4,search_rollouts=4,search_fraction=.5)
    assert env.policies.count("search")==2
    assert bool(env.legal_mask().any(1).all())
