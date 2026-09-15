import torch
from flysans.model import RecurrentPolicy
from flysans.topology_controls import make_control

def test_recurrent_baselines_share_policy_interface():
    for cell in ("gru","lstm","rnn"):
        model=RecurrentPolicy(154,38,32,cell);state=model.initial_state(4);logits,value,next_state=model(torch.randn(4,154),state)
        assert logits.shape==(4,38);assert value.shape==(4,);assert next_state.shape==state.shape

def test_degree_rewire_preserves_directed_degree_multisets():
    graph={"neuron_count":4,"indices":torch.tensor([[0,0,1,2,3],[1,2,2,3,0]]),"weights":torch.ones(5),"edge_groups":torch.zeros(5,dtype=torch.long)}
    control=make_control(graph,"degree_rewire",7)
    assert torch.equal(torch.bincount(graph["indices"][0],minlength=4),torch.bincount(control["indices"][0],minlength=4))
    assert torch.equal(torch.bincount(graph["indices"][1],minlength=4),torch.bincount(control["indices"][1],minlength=4))
