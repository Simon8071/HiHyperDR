from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool
from torch.nn import Sequential, Linear, ReLU


class DrugEncoder(nn.Module):
    def __init__(self, d_atom, d_model=128, dropout=0.2, hidden_dim=32, num_total_atoms=None):
        super().__init__()
        self.d_model = d_model
        dim = hidden_dim
        self.dropout_ratio = dropout
        nn1 = Sequential(Linear(d_atom, dim), ReLU(), Linear(dim, dim))
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(dim)
        nn2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv2 = GINConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(dim)
        nn3 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv3 = GINConv(nn3)
        self.bn3 = torch.nn.BatchNorm1d(dim)
        nn4 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv4 = GINConv(nn4)
        self.bn4 = torch.nn.BatchNorm1d(dim)
        nn5 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv5 = GINConv(nn5)
        self.bn5 = torch.nn.BatchNorm1d(dim)
        self.fc1_xd = Linear(dim, d_model)
        if num_total_atoms is not None:
            self.atom_mask = nn.Parameter(torch.ones(num_total_atoms, 1))

    def forward(self, batched_data):
        x, edge_index, batch = batched_data.x, batched_data.edge_index, batched_data.batch

        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x = F.relu(self.conv4(x, edge_index))
        x = self.bn4(x)
        x = F.relu(self.conv5(x, edge_index))
        x = self.bn5(x)
        if hasattr(self, 'atom_mask'):
            x = x * self.atom_mask
        x = global_add_pool(x, batch)

        x = F.relu(self.fc1_xd(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        return x


class GraphCDRMutationEncoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.cov1 = nn.Conv2d(1, 50, (1, 700), stride=(1, 5))
        self.cov2 = nn.Conv2d(50, 30, (1, 5), stride=(1, 2))

        def get_conv_out_dim(dim_in, kernel_size, stride, padding=0):
            return (dim_in + 2 * padding - kernel_size) // stride + 1

        w1 = get_conv_out_dim(input_dim, 700, 5)
        w2 = w1 // 5

        w3 = get_conv_out_dim(w2, 5, 2)
        w4 = w3 // 10

        self.flatten_dim = 30 * w4
        self.fc_mut = nn.Linear(self.flatten_dim, output_dim)

    def forward(self, x):
        x = x.unsqueeze(1).unsqueeze(2)
        x = torch.tanh(self.cov1(x))
        x = F.max_pool2d(x, (1, 5))
        x = F.relu(self.cov2(x))
        x = F.max_pool2d(x, (1, 10))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc_mut(x))
        return x


class GraphCDROmicsEncoder(nn.Module):
    def __init__(self, omics_dims: List[int], output_dim=128, num_cells=None):
        super().__init__()
        self.num_omics = len(omics_dims)
        self.omics_encoders = nn.ModuleList()
        if num_cells is not None:
            self.gene_mask = nn.Parameter(torch.ones(num_cells, omics_dims[0]))
        for idx, dim in enumerate(omics_dims):
            if idx == 1:
                self.omics_encoders.append(GraphCDRMutationEncoder(dim, output_dim))
            else:
                encoder = nn.Sequential(
                    nn.Linear(dim, 256),
                    nn.BatchNorm1d(256),
                    nn.Tanh(),
                    nn.Linear(256, output_dim),
                    nn.ReLU()
                )
                self.omics_encoders.append(encoder)

        self.fusion = nn.Sequential(
            nn.Linear(output_dim * self.num_omics, output_dim),
            nn.ReLU()
        )

    def forward(self, omics_list: List[torch.Tensor]):
        encoded_feats = []
        for i, data in enumerate(omics_list):
            if i == 0 and hasattr(self, 'gene_mask'):
                data = data * self.gene_mask
            encoded_feats.append(self.omics_encoders[i](data))
        x_concat = torch.cat(encoded_feats, dim=1)
        x_cell = self.fusion(x_concat)

        return x_cell
