from typing import List, Tuple

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pyparsing import Optional
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj
from torch_geometric.utils.convert import from_networkx


def train_gcn(model: nn.Module, data: Data, opt: torch.optim.Optimizer, gcn_lambda: float = 10.0, n_epochs: int = 1):
    """Trains GNN `model` for `n_epochs`

    Args:
        model (nn.Module): the model to train
        data (Data): torch_geometric Data object
        opt (torch.optim.Optimizer): model optimizer
        gcn_lambda (float, optional): mixing coefficient between GNN losses. Defaults to 10..
        n_epochs (int, optional): the number of epochs to train. Defaults to 1.
    """
    assert data.num_nodes > 1

    adj = to_dense_adj(data.edge_index).squeeze()
    deg = torch.diag(adj.sum(1).squeeze())
    laplacian = deg - adj

    model.train()
    for epoch in range(n_epochs):
        opt.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        soft_out = torch.unsqueeze(torch.nn.functional.softmax(out, dim=1)[:, 1], 1)
        loss_reg = torch.mm(torch.mm(soft_out.T, laplacian), soft_out)
        loss += gcn_lambda * loss_reg.squeeze()
        loss.backward()
        opt.step()


def create_pyg_data(graph: nx.Graph, features: torch.Tensor, rew_states: List[Tuple[int, float]] = None) -> Data:
    """Creates torch_geometric.data.Data instance

    Args:
        graph (nx.Graph): input graph
        features (torch.Tensor): node feature matrix with shape [num_nodes, num_node_features]
        rew_states (List[Tuple[int, float]], optional): list of states with non-zero reward. Defaults to None.

    Returns:
        Data: resulting torch_geometric.data.Data instance
    """
    labels = torch.zeros(features.shape[0], dtype=torch.long)
    data = from_networkx(graph)
    data.x = features
    data.y = labels
    if rew_states is not None:
        train_mask = torch.zeros(features.shape[0], dtype=torch.long)
        for idx, r in rew_states:
            labels[idx] = 1 if r > 0.0 else 0
            train_mask[idx] = 1
        data.train_mask = train_mask

    return data
