import torch
import torch.nn as nn
import torch.nn.functional as F
import math
class GCN(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GCN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GCNModel(nn.Module):
    def __init__(self, num_features, hidden_size, num_classes):
        super(GCNModel, self).__init__()
        self.gc1 = GCN(num_features, hidden_size)
        self.gc2 = GCN(hidden_size, num_classes)

    def forward(self, inputs, adj):
        x = F.relu(self.gc1(inputs, adj))
        x = self.gc2(x, adj)
        return x
