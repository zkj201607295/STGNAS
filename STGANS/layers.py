from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
import torch
import math
import torch.nn.functional as F
from torch import nn
from torch.nn import Linear
from Grapro import Gra_inc
from gat_conv import GATConv
from torch_geometric.nn import GCNConv


class GraphConvolutionBS(Module):
    def __init__(self, in_features, out_features, activation=lambda x: x, withbn=False, withloop=False, bias=True,
                 res=False):
        super(GraphConvolutionBS, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma = activation
        self.res = res
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.conv = GCNConv(in_features, out_features)

        if withloop:
            self.self_weight = Parameter(torch.FloatTensor(in_features, out_features))
        else:
            self.register_parameter("self_weight", None)

        if withbn:
            self.bn = torch.nn.BatchNorm1d(out_features)
        else:
            self.register_parameter("bn", None)

        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size()[1])
        self.weight.data.uniform_(-stdv, stdv)
        if self.self_weight is not None:
            stdv = 1. / math.sqrt(self.self_weight.size()[1])
            self.self_weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        #support = torch.mm(input, self.weight)
        output = self.conv(input, adj)
        #print(output)
        #output = torch.spmm(adj, support)

        # Self-loop
        if self.self_weight is not None:
            output = output + torch.mm(input, self.self_weight)

        if self.bias is not None:
            output = output + self.bias
        # BN
        if self.bn is not None:
            output = self.bn(output)
        # Res
        if self.res:
            return self.sigma(output) + input
        else:
            return self.sigma(output)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + str(self.in_features) + ', ' \
               + str(self.out_features) + ')'


class JKNetBlock(Module):
    def __init__(self, in_channels, out_channels, nlayers, withbn=False, withloop=False, activation=F.relu,
                 dropout=0.5, aggrmethod='concat', dense=True):
        super(JKNetBlock, self).__init__()
        self.in_features = in_channels
        self.out_features = out_channels
        self.hiddendim = out_channels
        self.nhiddenlayer = nlayers
        self.activation = activation
        self.aggrmethod = aggrmethod
        self.dense = dense
        self.dropout = dropout
        self.withbn = withbn
        self.withloop = withloop

        self.hiddenlayers = nn.ModuleList()
        for i in range(self.nhiddenlayer):
            if i == 0:
                layer = GraphConvolutionBS(self.in_features, self.hiddendim, self.activation, self.withbn,
                                           self.withloop)
            else:
                layer = GraphConvolutionBS(self.hiddendim, self.hiddendim, self.activation, self.withbn, self.withloop)
            self.hiddenlayers.append(layer)

    def _doconcat(self, x, subx):
        if x is None:
            return subx
        if self.aggrmethod == "concat":
            return torch.cat((x, subx), 1)
        elif self.aggrmethod == "add":
            return x + subx
        elif self.aggrmethod == "nores":
            return x

    def forward(self, x, adj, weight):
        jknetout = [x]
        for gconv in self.hiddenlayers:
            x = gconv(x, adj)
            x = F.dropout(x, self.dropout, training=self.training)
            jknetout.append(x)
        x = torch.stack(jknetout, dim=0)
        return torch.max(x, dim=0)[0]

    def get_outdim(self):
        return self.out_features


class InceptionGCNBlock(Module):
    def __init__(self, in_channels, out_channels, nlayers, withbn=False, withloop=False, activation=F.relu,
                 dropout=0.5, aggrmethod='concat', dense=True):
        super(InceptionGCNBlock, self).__init__()
        self.in_features = in_channels
        self.reduce_channels = in_channels + out_channels * nlayers
        self.out_features = out_channels
        self.hiddendim = out_channels
        self.nhiddenlayer = nlayers
        self.activation = activation
        self.aggrmethod = aggrmethod
        self.dense = dense
        self.dropout = dropout
        self.withbn = withbn
        self.withloop = withloop

        self.hiddenlayers = nn.ModuleList()
        for i in range(self.nhiddenlayer):
            layers = nn.ModuleList()
            for j in range(i + 1):
                if j == 0:
                    layer = GraphConvolutionBS(self.in_features, self.hiddendim, self.activation, self.withbn,
                                               self.withloop)
                else:
                    layer = GraphConvolutionBS(self.hiddendim, self.hiddendim, self.activation, self.withbn,
                                               self.withloop)
                layers.append(layer)
            self.hiddenlayers.append(layers)
        self.hiddenlayers.append(GraphConvolutionBS(self.reduce_channels, self.out_features, self.activation,
                                                    self.withbn, self.withloop))

    def _doconcat(self, x, subx):
        if self.aggrmethod == 'concat':
            return torch.cat((x, subx), 1)
        elif self.aggrmethod == 'add':
            return x + subx

    def forward(self, input, adj, weight):
        x = input
        for layers in self.hiddenlayers[:-1]:
            subx = input
            for gconv in layers:
                subx = gconv(subx, adj)
                subx = F.dropout(subx, self.dropout, training=self.training)
            x = self._doconcat(x, subx)
        return F.dropout(self.hiddenlayers[self.nhiddenlayer](x, adj), self.dropout, training=self.training)

    def get_outdim(self):
        return self.out_features


class ResGCNBlock(Module):
    def __init__(self, in_channels, out_channels, nlayers, withbn=False, withloop=False, activation=F.relu,
                 dropout=0.5, aggrmethod=None, dense=None):
        super(ResGCNBlock, self).__init__()
        self.in_features = in_channels
        self.reduce_channels = in_channels + out_channels * nlayers
        self.out_features = out_channels
        self.hiddendim = out_channels
        self.nhiddenlayer = nlayers
        self.activation = activation
        self.aggrmethod = aggrmethod
        self.dense = dense
        self.dropout = dropout
        self.withbn = withbn
        self.withloop = withloop

        self.hiddenlayers = nn.ModuleList()
        for i in range(self.nhiddenlayer):
            if i == 0:
                layer = GraphConvolutionBS(self.in_features, self.hiddendim, self.activation, self.withbn,
                                           self.withloop, res=True)
            else:
                layer = GraphConvolutionBS(self.hiddendim, self.hiddendim, self.activation, self.withbn,
                                           self.withloop, res=True)
            self.hiddenlayers.append(layer)

    def forward(self, input, adj, weight):
        x = input
        for gconv in self.hiddenlayers:
            x = gconv(x, adj)
            x = F.dropout(x, self.dropout, training=self.training)
        return x

    def get_outdim(self):
        return self.model.get_outdim()


class DenseJKNetBlock(Module):
    def __init__(self, in_channels, out_channels, nlayers, withbn=False, withloop=False, activation=F.relu,
                 dropout=0.5, aggrmethod='concat', dense=True):
        super(DenseJKNetBlock, self).__init__()
        self.in_features = in_channels
        self.out_features = out_channels
        self.reduce_channels = in_channels + out_channels * nlayers
        self.hiddendim = out_channels
        self.nhiddenlayer = nlayers
        self.activation = activation
        self.aggrmethod = aggrmethod
        self.dense = dense
        self.dropout = dropout
        self.withbn = withbn
        self.withloop = withloop

        self.hiddenlayers = nn.ModuleList()
        for i in range(self.nhiddenlayer):
            if i == 0:
                layer = GraphConvolutionBS(self.in_features, self.hiddendim, self.activation, self.withbn,
                                           self.withloop)
            else:
                layer = GraphConvolutionBS(self.hiddendim, self.hiddendim, self.activation, self.withbn, self.withloop)
            self.hiddenlayers.append(layer)
        self.hiddenlayers.append(GraphConvolutionBS(self.reduce_channels, self.hiddendim, self.activation, self.withbn, self.withloop))

    def _doconcat(self, x, subx):
        if x is None:
            return subx
        if self.aggrmethod == "concat":
            return torch.cat((x, subx), 1)
        elif self.aggrmethod == "add":
            return x + subx
        elif self.aggrmethod == "nores":
            return x

    def forward(self, input, adj, weight):
        x = input
        subx = input
        for gconv in self.hiddenlayers[:-1]:
            subx = gconv(subx, adj)
            subx = F.dropout(subx, self.dropout, training=self.training)
            x = self._doconcat(x, subx)
        return F.dropout(self.hiddenlayers[self.nhiddenlayer](x, adj), self.dropout, training=self.training)


class ConvG(torch.nn.Module):
    def __init__(self, nlayers, activation=F.relu, dropout=0.5):
        super(ConvG, self).__init__()

        self.prop = Gra_inc(K=nlayers)
        self.activation = activation
        self.dropout = dropout


    def reset_parameters(self):
        self.prop.reset_parameters()

    def forward(self, x, edge_index):
        x = self.activation(x)
        x = self.prop(x, edge_index)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x

class GAT_Net(torch.nn.Module):
    def __init__(self, nlayers, in_channels, out_channels, activation=F.relu,dropout=0.5):
        super(GAT_Net, self).__init__()
        self.prop = GATConv(in_channels=in_channels, out_channels=out_channels, K = nlayers)
        self.dropout = dropout
        self.activation = activation

    def reset_parameters(self):
        self.prop.reset_parameters()

    def forward(self, x, edge_index):
        x = self.activation(x)
        x = self.prop(x, edge_index)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class MLP_Net(torch.nn.Module):
    def __init__(self, in_channels, out_channels, activation=F.relu, dropout=0.5):
        super(MLP_Net, self).__init__()
        self.line = Linear(in_features=in_channels, out_features=out_channels)
        self.dropout = dropout
        self.activation = activation

    def reset_parameters(self):
        self.line.reset_parameters()

    def forward(self, x, edge_index):
        x = self.activation(x)
        x = self.line(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x