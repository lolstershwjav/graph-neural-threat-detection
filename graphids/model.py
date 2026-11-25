"""Graph message-passing layers and joint flow and incident-host readouts."""

import torch
from torch import nn


def degree(index, n, dtype):
    return torch.bincount(index, minlength=n).to(dtype).clamp(min=1)


def receive(values, dst, n):
    return values.new_zeros((n, values.shape[1])).index_add_(0, dst, values)


def neighborhood(edges, n, topology, generator):
    if topology == "self":
        source = dest = torch.empty(0, dtype=torch.int64)
    else:
        source, dest = edges
        if topology == "rewired":
            dest = dest[torch.randperm(len(dest), generator=generator)]
    self_ids = torch.arange(n)
    # Unique ordered edges keep a repeated endpoint pair from dominating.
    pairs = torch.stack([torch.cat([source, dest, self_ids]),
                         torch.cat([dest, source, self_ids])])
    return torch.unique(pairs, dim=1)


class Layer(nn.Module):
    def __init__(self, size, method):
        super().__init__()
        self.method = method
        self.linear = nn.Linear(size * (2 if method == "sage" else 1), size)
        if method == "gat":
            self.attn_s = nn.Linear(size, 1, bias=False)
            self.attn_d = nn.Linear(size, 1, bias=False)
        self.norm = nn.LayerNorm(size)

    def forward(self, x, edges):
        src, dst = edges
        n = len(x)
        d = degree(dst, n, x.dtype)
        if self.method == "sage":
            # GraphSAGE mean includes the guaranteed self edge for every node.
            density = receive(x[src], dst, n) / d[:, None]
            mixed = self.linear(torch.cat([x, density], dim=1))
        else:
            h = self.linear(x)
            if self.method == "gat":
                unnormalized = torch.nn.functional.leaky_relu(
                    (self.attn_s(h).ravel()[src] + self.attn_d(h).ravel()[dst]), .2
                )
                maxima = h.new_full((n,), float("-inf"))
                maxima.scatter_reduce_(0, dst, unnormalized.detach(), "amax")
                weights = (unnormalized - maxima[dst]).exp()
                totals = h.new_zeros(n).index_add_(0, dst, weights)
                weights = weights / totals[dst]
            else:
                weights = (d[src] * d[dst]).rsqrt()
            mixed = receive(h[src] * weights[:, None], dst, n)
        return torch.relu(self.norm(mixed + x))


class Detector(nn.Module):
    """Read out each directed flow and node incidence from anonymous node features."""

    def __init__(self, feature_count, flow_count, width, method, temporal=False, auxiliary=True):
        super().__init__()
        self.temporal = temporal
        self.auxiliary = auxiliary
        self.encoder = nn.Linear(feature_count, width)
        if temporal:
            self.gru = nn.GRU(feature_count + 1, width, batch_first=True)
        self.layers = nn.ModuleList([Layer(width, method), Layer(width, method)])
        self.flow_head = nn.Sequential(nn.Linear(2 * width + flow_count, width),
                                       nn.ReLU(), nn.Dropout(.1), nn.Linear(width, 1))
        self.host_head = nn.Linear(width, 1)

    def forward(self, x, edges, flow, history, graph):
        current = torch.relu(self.encoder(x))
        if self.temporal:
            _, state = self.gru(history)
            current = current + state[0]
        for layer in self.layers:
            current = layer(current, graph)
        src, dst = edges
        pair = torch.cat([current[src], current[dst], flow], dim=1)
        return self.flow_head(pair).ravel(), self.host_head(current).ravel()
