import torch
from torch.nn import Linear, BatchNorm1d, Module
from torch_geometric.nn import SAGEConv, global_max_pool
import torch.nn.functional as F

class GCN(Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = SAGEConv(768, hidden_channels)
        self.bn1 = BatchNorm1d(hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.bn2 = BatchNorm1d(hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels)
        self.bn3 = BatchNorm1d(hidden_channels)
        self.lin = Linear(hidden_channels, 11)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = global_max_pool(x, batch)
        embeddings = x  # Optional: return embeddings if needed for other tasks

        x = F.dropout(x, p=0.1675, training=self.training)
        x = self.lin(x)
        return x