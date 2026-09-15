import torch
from torch import nn
from torch.nn import functional as F

from .connectome import SparseGraph


class _SparsePlasticMM(torch.autograd.Function):
    """Sparse SpMM whose edge gradient never materializes a dense NxN matrix."""
    @staticmethod
    def forward(ctx, indices, weights, state, size, window_start, window_count):
        adjacency=torch.sparse_coo_tensor(indices,weights,(size,size),is_coalesced=True,check_invariants=False)
        ctx.save_for_backward(indices,weights,state);ctx.size=size;ctx.window_start=window_start;ctx.window_count=window_count
        return torch.sparse.mm(adjacency,state.transpose(0,1)).transpose(0,1)

    @staticmethod
    def backward(ctx, grad_output):
        indices,weights,state=ctx.saved_tensors
        adjacency=torch.sparse_coo_tensor(indices,weights,(ctx.size,ctx.size),is_coalesced=True,check_invariants=False)
        grad_state=torch.sparse.mm(adjacency.transpose(0,1),grad_output.transpose(0,1)).transpose(0,1)
        grad_weights=torch.zeros_like(weights);edge_count=weights.numel();remaining=min(ctx.window_count,edge_count);position=ctx.window_start%edge_count;chunk=16384
        while remaining:
            take=min(remaining,edge_count-position,chunk);sl=slice(position,position+take);rows=indices[0,sl];cols=indices[1,sl]
            grad_weights[sl]=(grad_output[:,rows]*state[:,cols]).sum(0)
            position=(position+take)%edge_count;remaining-=take
        return None,grad_weights,grad_state,None,None,None


class FlyPolicy(nn.Module):
    def __init__(self, graph: SparseGraph, observation_size: int, action_size: int, plastic_edges: bool = False):
        super().__init__()
        graph.validate()
        self.neuron_count = graph.neuron_count
        self.register_buffer("edge_indices", graph.indices.long(), persistent=False)
        self.register_buffer("edge_weights", graph.weights.float(), persistent=False)
        self.plastic_edges = plastic_edges
        if plastic_edges:
            self.source_gain_logits = nn.Parameter(torch.zeros(graph.neuron_count, dtype=torch.float32, device=graph.weights.device))
            self.target_gain_logits = nn.Parameter(torch.zeros(graph.neuron_count, dtype=torch.float32, device=graph.weights.device))
        self.edge_window_start = 0
        self.edge_window_count = graph.weights.numel()
        groups = graph.edge_groups if graph.edge_groups is not None else (graph.weights < 0).long()
        self.register_buffer("edge_groups", groups.long(), persistent=False)
        adjacency = torch.sparse_coo_tensor(
            graph.indices, graph.weights,
            (graph.neuron_count, graph.neuron_count), check_invariants=False,
        ).coalesce()
        self.register_buffer("adjacency", adjacency, persistent=False)
        neuron_groups = torch.full((graph.neuron_count,), 2, dtype=torch.long, device=groups.device)
        neuron_groups.scatter_(0, graph.indices[1].long(), groups.long())
        self.register_buffer("neuron_groups", neuron_groups, persistent=False)
        self.encoder = nn.Linear(observation_size, graph.neuron_count, bias=False)
        self.synaptic_gain = nn.Parameter(torch.tensor([0.5413, 0.5413, 0.0]))
        self.leak_logit = nn.Parameter(torch.tensor(0.0))
        self.policy_head = nn.Linear(graph.neuron_count, action_size)
        self.value_head = nn.Linear(graph.neuron_count, 1)

    def effective_edge_weights(self):
        if not self.plastic_edges:
            return self.edge_weights
        source=0.5+torch.sigmoid(self.source_gain_logits)
        target=0.5+torch.sigmoid(self.target_gain_logits)
        return self.edge_weights*source[self.edge_indices[1]]*target[self.edge_indices[0]]

    def plasticity_penalty(self):
        if not self.plastic_edges:
            return self.edge_weights.new_zeros(())
        return .5*((torch.sigmoid(self.source_gain_logits)-.5).square().mean()+(torch.sigmoid(self.target_gain_logits)-.5).square().mean())

    def set_plasticity_window(self, update: int, fraction: float):
        # Factorized plasticity updates every effective edge on every step.
        self.edge_window_count=self.edge_weights.numel();self.edge_window_start=0

    def initial_state(self, batch_size: int, device=None) -> torch.Tensor:
        return torch.zeros(batch_size, self.neuron_count, device=device or self.edge_weights.device)

    def forward(self, observation: torch.Tensor, state: torch.Tensor):
        gains = torch.stack((F.softplus(self.synaptic_gain[0]), F.softplus(self.synaptic_gain[1]), torch.tanh(self.synaptic_gain[2])))
        scaled_state = state * gains[self.neuron_groups]
        if self.plastic_edges:
            scaled_state=scaled_state*(0.5+torch.sigmoid(self.source_gain_logits))
        adjacency = self.adjacency
        recurrent = torch.sparse.mm(adjacency, scaled_state.transpose(0, 1)).transpose(0, 1)
        if self.plastic_edges:
            recurrent=recurrent*(0.5+torch.sigmoid(self.target_gain_logits))
        candidate = torch.tanh(recurrent + self.encoder(observation))
        leak = torch.sigmoid(self.leak_logit)
        next_state = leak * state + (1.0 - leak) * candidate
        return self.policy_head(next_state), self.value_head(next_state).squeeze(-1), next_state


class DeepFlyPolicy(FlyPolicy):
    """MaleCNS controller plus a compact nonlinear memory for card strategy."""
    def __init__(self, graph: SparseGraph, observation_size: int, action_size: int, decision_size: int = 256, plastic_edges: bool = False):
        super().__init__(graph, observation_size, action_size, plastic_edges)
        self.brain_size = self.neuron_count
        self.decision_size = decision_size
        self.neuron_count = self.brain_size + decision_size
        self.decision_gru = nn.GRUCell(observation_size + action_size + 1, decision_size)
        self.deep_policy = nn.Sequential(nn.LayerNorm(decision_size), nn.Linear(decision_size, decision_size), nn.GELU(), nn.Linear(decision_size, action_size))
        self.deep_value = nn.Sequential(nn.LayerNorm(decision_size), nn.Linear(decision_size, decision_size // 2), nn.GELU(), nn.Linear(decision_size // 2, 1))
        nn.init.zeros_(self.deep_policy[-1].weight); nn.init.zeros_(self.deep_policy[-1].bias)
        nn.init.zeros_(self.deep_value[-1].weight); nn.init.zeros_(self.deep_value[-1].bias)

    def initial_state(self, batch_size: int, device=None):
        return torch.zeros(batch_size, self.neuron_count, device=device or self.edge_weights.device)

    def forward(self, observation: torch.Tensor, state: torch.Tensor):
        brain_state=state[:,:self.brain_size]; decision_state=state[:,self.brain_size:]
        base_logits,base_value,next_brain=super().forward(observation,brain_state)
        decision_input=torch.cat((observation,base_logits,base_value[:,None]),dim=-1)
        next_decision=self.decision_gru(decision_input,decision_state)
        logits=base_logits+self.deep_policy(next_decision)
        value=base_value+self.deep_value(next_decision).squeeze(-1)
        return logits,value,torch.cat((next_brain,next_decision),dim=-1)


class MLPPolicy(nn.Module):
    def __init__(self, observation_size: int, action_size: int, hidden_size: int = 256):
        super().__init__()
        self.neuron_count = hidden_size
        self.backbone = nn.Sequential(nn.Linear(observation_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, hidden_size), nn.Tanh())
        self.policy_head = nn.Linear(hidden_size, action_size)
        self.value_head = nn.Linear(hidden_size, 1)

    def initial_state(self, batch_size: int, device=None):
        return torch.zeros(batch_size, self.neuron_count, device=device)

    def forward(self, observation, state):
        hidden = self.backbone(observation)
        return self.policy_head(hidden), self.value_head(hidden).squeeze(-1), hidden


class RecurrentPolicy(nn.Module):
    """Shared GRU/LSTM/Elman baseline interface for controlled experiments."""
    def __init__(self, observation_size: int, action_size: int, hidden_size: int = 1024, cell: str = "gru"):
        super().__init__(); self.hidden_size=hidden_size; self.cell_type=cell
        if cell=="gru": self.cell=nn.GRUCell(observation_size,hidden_size); self.state_parts=1
        elif cell=="lstm": self.cell=nn.LSTMCell(observation_size,hidden_size); self.state_parts=2
        elif cell=="rnn": self.cell=nn.RNNCell(observation_size,hidden_size,nonlinearity="tanh"); self.state_parts=1
        else: raise ValueError(f"unknown recurrent cell: {cell}")
        self.neuron_count=hidden_size*self.state_parts
        self.policy_head=nn.Sequential(nn.LayerNorm(hidden_size),nn.Linear(hidden_size,hidden_size//2),nn.GELU(),nn.Linear(hidden_size//2,action_size))
        self.value_head=nn.Sequential(nn.LayerNorm(hidden_size),nn.Linear(hidden_size,hidden_size//2),nn.GELU(),nn.Linear(hidden_size//2,1))

    def initial_state(self,batch_size:int,device=None):
        return torch.zeros(batch_size,self.neuron_count,device=device)

    def forward(self,observation,state):
        if self.cell_type=="lstm":
            h,c=state.split(self.hidden_size,dim=-1); next_h,next_c=self.cell(observation,(h,c)); next_state=torch.cat((next_h,next_c),-1)
        else: next_h=self.cell(observation,state); next_state=next_h
        return self.policy_head(next_h),self.value_head(next_h).squeeze(-1),next_state
