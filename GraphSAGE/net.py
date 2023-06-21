#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init # 用于初始化神经网络

'''
s1: 邻居节点汇聚
s2：中心节点结合邻居节点进行更新
s3: 汇总邻居采样以及邻居聚合操作组合图神经网络
'''

class NeighborAggregator(nn.Module):
    '''
    input_dim: 输入特征维度
    output_dim: 输出特征维度
    use_bias: 是否使用偏置
    aggr_method: 邻居聚合方式
    '''
    def __init__(self, input_dim, output_dim, use_bias=False, aggr_method='mean'):
        super(NeighborAggregator, self).__init__()
        self.input_dim=input_dim
        self.output_dim=output_dim
        self.use_bias=use_bias
        self.aggr_method=aggr_method
        self.weight=nn.Parameter(torch.Tensor(self.input_dim, self.output_dim))
        if self.use_bias:
            self.bias=nn.Parameter(torch.Tensor(self.output_dim))
        self.reset_parameters()
    
    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)
        if self.use_bias:
            init.zeros_(self.bais)

    def forward(self, neighbor_feature):
        if self.aggr_method=='mean':
            aggr_neighbor=neighbor_feature.mean(dim=1)
        elif self.aggr_method=='sum':
            aggr_neighbor=neighbor_feature.sum(dim=1)
        elif self.aggr_method=='max':
            aggr_neighbor=neighbor_feature.max(dim=1)
        else:
            raise ValueError("Unkonw aggr type, expected sum, max, or mean, but got {}".format(self.aggr_method))        

        neighbor_hidden=torch.matmul(aggr_neighbor, self.weight)
        if self.use_bias:
            neighbor_hidden += self.bias
        return neighbor_hidden
    
    def extra_repr(self):
        return 'in_features={}, out_features={}, aggr_method={}'.format(self.input_dim, self.output_dim, self.aggr_method)

class SageGCN(nn.Module):
    '''
    0.定义： 实现中心节点与汇聚后邻居节点的结合
    '''
    def __init__(self, input_dim, hidden_dim,activation=F.relu,aggr_neighbor_method='mean',aggr_hidden_method='sum'):
        super(SageGCN, self).__init__()
        self.input_dim=input_dim
        self.hidden_dim=hidden_dim
        self.activation=activation
        self.aggr_neighbor_method=aggr_neighbor_method
        self.aggr_hidden_method=aggr_hidden_method
        self.aggregator=NeighborAggregator(self.input_dim, self.hidden_dim, aggr_method=self.aggr_neighbor_method)
        self.weight=nn.Parameter(torch.Tensor(input_dim,hidden_dim))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform(self.weight)
        self.aggregator.reset_parameters()
    
    def forward(self, src_node_features, neighbor_node_features):
        neighbor_hidden=self.aggregator(neighbor_node_features)
        self_hidden=torch.matmul(src_node_features, self.weight)
        if self.aggr_hidden_method=='sum':
            hidden=self_hidden+neighbor_hidden
        elif self.aggr_hidden_method=='contact':
            hidden=torch.cat([self_hidden, neighbor_hidden],dim=1)
        else:
            raise ValueError("Expected sum or concat, got {}"
                             .format(self.aggr_hidden))
        if self.activation:
            return self.activation(hidden)
        else:
            return hidden

    def extra_repr(self):
        output_dim = self.hidden_dim if self.aggr_hidden_method == "sum" else self.hidden_dim * 2
        return 'in_features={}, out_features={}, aggr_hidden_method={}'.format(
            self.input_dim, output_dim, self.aggr_hidden_method)

class GraphSage(nn.Module):
    '''
    0: 定义一个两层/两阶段的模型，隐藏层节点数为64，每阶段的采样数目为10，返回中心节点的输出
    1: node_features_list: 第0个元素表示源节点的特征，其后元素表示每阶段采样得到的邻居特征
    '''
    def __init__(self, input_dim, hidden_dim, num_neighbors_list):
        super(GraphSage, self).__init__()
        self.input_dim=input_dim
        self.num_neighbors_list=num_neighbors_list
        self.num_layers=len(num_neighbors_list)
        self.gcn=nn.ModuleList()
        self.gcn.append(SageGCN(input_dim, hidden_dim[0]))
        for index in range(0, len(hidden_dim) - 2):
            self.gcn.append(SageGCN(hidden_dim[index], hidden_dim[index+1]))
        self.gcn.append(SageGCN(hidden_dim[-2], hidden_dim[-1], activation=None))
    def forward(self, node_features_list):
        '''target: src_node_features, neighbors_node_features'''
        hidden = node_features_list
        for l in range(self.num_layers): # l= 0 [hop=0 ,1 ], 1 [hop=0]
            next_hidden = []
            gcn = self.gcn[l]
            for hop in range(self.num_layers - l): 
                src_node_features = hidden[hop]
                src_node_num = len(src_node_features)
                neighbor_node_features = hidden[hop + 1] \
                    .view((src_node_num, self.num_neighbors_list[hop], -1))
                h = gcn(src_node_features, neighbor_node_features)
                next_hidden.append(h)
            hidden = next_hidden
        return hidden[0]

    def extra_repr(self):
        return 'in_features={}, num_neighbors_list={}'.format(
            self.input_dim, self.num_neighbors_list
        )
