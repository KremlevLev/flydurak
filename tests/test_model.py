import torch

from flysans.connectome import make_synthetic_graph
from flysans.model import FlyPolicy


def test_plastic_edges_start_at_biological_weights_and_receive_gradients():
    from flysans.connectome import make_synthetic_graph
    graph=make_synthetic_graph(32,4,7)
    model=FlyPolicy(graph,10,5,plastic_edges=True)
    assert torch.allclose(model.effective_edge_weights(),graph.weights)
    state=model.initial_state(3);obs=torch.randn(3,10)
    logits,value,_=model(obs,state)
    (logits.square().mean()+value.square().mean()).backward()
    assert model.source_gain_logits.grad is not None
    assert model.target_gain_logits.grad is not None
    assert torch.isfinite(model.source_gain_logits.grad).all()
    assert torch.isfinite(model.target_gain_logits.grad).all()
    assert model.plasticity_penalty().item()==0.0


def test_policy_shapes_and_gradient_flow():
    graph = make_synthetic_graph(32, 3, seed=2)
    model = FlyPolicy(graph, observation_size=8, action_size=5)
    state = model.initial_state(5)
    logits, value, next_state = model(torch.randn(5, 8), state)
    assert logits.shape == (5, 5)
    assert value.shape == (5,)
    assert next_state.shape == (5, 32)
    (logits.mean() + value.mean()).backward()
    assert model.synaptic_gain.grad is not None
    assert torch.equal(model.edge_indices, graph.indices)
