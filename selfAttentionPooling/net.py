'''根据提出的两个网络架构图，现网络需要基础单元如下:
1. 图卷积层： GraphConvolution
2. 图池化层： self-attention pooling
3. 读出机制： readout: global_max_pool; global_avg_pool;
----------->利用上述原件组合成两个网络架构图<------------
'''

#%%
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.nn.init as init
import torch_scatter
import scipy.sparse as sp
from utilits_function import *
#%%
# define the GraphConvolution
class GraphConvolution(nn.Module):
    def __init__(self, input_dim, out_dim, use_bias=False):
        super(GraphConvolution,self).__init__()
        # LXW
        self.input_dim=input_dim
        self.out_dim=out_dim
        self.weight=nn.Parameter(torch.Tensor(self.input_dim, self.out_dim))
        self.use_bias=use_bias
        if self.use_bias:
            self.bias=nn.Parameter(torch.Tensor(self.out_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    
    def reset_parameters(self):
        init.kaiming_uniform(self.weight)
        if self.bias:
            init.kaiming_uniform(self.bias)
    
    def forward(self, adjacency, input_feature):
        support=torch.mm(input_feature,self.weight)
        output=torch.sparse.mm(adjacency, support)
        if self.use_bias:
            output += self.bias
        return output
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.input_dim) + ' -> ' \
            + str(self.output_dim) + ')'

# define pooling layer based on self-attention
'''
1. 计算节点重要度分数-----------------------------------> GraphConvolution()
2. 根据节点重要度分数和拓扑结构，丢弃一些不重要的节点------> top_rank()
3. 更新邻接矩阵和节点的特征----->得到池化结果-------------> filter_adjacency
'''

def top_rank(attention_score, graph_indicator, keep_ratio):
    '''
    目标： 基于给定的attention_score， 对每个图进行pooling操作
           对每个图进行池化，最后将其级联起来进行下一步计算
    attention_score: torch.Tensor() 使用GraphConvolution计算的图注意力分数
    graph_indicator: torch.Tensor() 指示每个节点属于哪个图
    keep_ratio: float 要保留的节点数 int(N * keep_ratio)
    '''
    graph_id_list = list(set(graph_indicator.cpu().numpy()))
    mask=attention_score.new_empty((0,), dtype=torch.bool)
    for graph_id in graph_id_list:
        graph_attn_score=attention_score[graph_indicator == graph_id]
        graph_node_num = len(graph_attn_score)
        graph_mask=attention_score.new_zeros((graph_node_num,),dtype=torch.bool)
        keep_graph_node_num=int(keep_ratio*graph_node_num)
        _, sorted_index = graph_attn_score.sort(descending=True)
        graph_mask[sorted_index[:keep_graph_node_num]]=True
        mask = torch.cat((mask, graph_mask))
    return mask

def filter_adjacency(adjacency, mask):
    '''
    根据mask对图结构进行更新

    Args:
        adjacency: torch.sparse.FloatTensor 池化之前的邻接矩阵
        mask: torch.Tensor (dtype= torch.bool) 节点掩码向量
    Returns:
        torch.sparse.FloatTensor, 池化之后归一化的邻接矩阵
    '''
    device = adjacency.device
    mask=mask.cpu().numpy
    indices= adjacency.coalesce().indices().cpu().numpy()
    num_nodes = adjacency.size(0)
    row, col = indices
    maskout_self_loop = row!=col
    row=row[maskout_self_loop]
    col=col[maskout_self_loop]
    sparse_adjacency = sp.csr_matrix((np.ones(len(row)), (row, col)),
                                     shape=(num_nodes, num_nodes), dtype=np.float32)
    #filtered_adjacency = sparse_adjacency[mask, :][:, mask]
    filtered_adjacency = sparse_adjacency
    return normalization(filtered_adjacency).to(device)

# readout 
def global_max_pool(x, graph_indicator):
    num = graph_indicator.max().item() + 1
    return torch_scatter.scatter_max(x, graph_indicator, dim=0, dim_size=num)[0]


def global_avg_pool(x, graph_indicator):
    num = graph_indicator.max().item() + 1
    return torch_scatter.scatter_mean(x, graph_indicator, dim=0, dim_size=num)

class SelfAttentionPooling(nn.Module):
    def __init__(self, input_dim, keep_ratio, activation=torch.tanh):
        super(SelfAttentionPooling, self).__init__()
        self.input_dim=input_dim
        self.keep_ratio=keep_ratio
        self.activation=activation
        self.attn_gcn=GraphConvolution(self.input_dim, 1)
    def forward(self, adjacency, input_feature, graph_indicator):
        attn_score = self.attn_gcn(adjacency, input_feature).squeeze()
        attn_score = self.activation(attn_score)
        
        mask = top_rank(attn_score, graph_indicator, self.keep_ratio)
        hidden = input_feature[mask] * attn_score[mask].view(-1, 1)
        mask_graph_indicator = graph_indicator[mask]
        mask_adjacency = filter_adjacency(adjacency, mask)
        return hidden, mask_graph_indicator, mask_adjacency

class modelA(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes=2):
        super(modelA, self).__init__()
        self.input_dim=input_dim
        self.hidden_dim=hidden_dim
        self.num_classes=num_classes

        self.gcn1=GraphConvolution(input_dim, hidden_dim)
        self.gcn2=GraphConvolution(hidden_dim, hidden_dim)
        self.gcn3=GraphConvolution(hidden_dim, hidden_dim)
        self.pool=SelfAttentionPooling(hidden_dim * 3, 0.5)
        self.fc1 = nn.Linear(hidden_dim * 3 * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, num_classes)

    def forward(self, adjacency, input_feature, graph_indicator):
        gcn1 = F.relu(self.gcn1(adjacency, input_feature))
        gcn2 = F.relu(self.gcn2(adjacency, gcn1))
        gcn3 = F.relu(self.gcn3(adjacency, gcn2))
        
        gcn_feature = torch.cat((gcn1, gcn2, gcn3), dim=1)
        pool, pool_graph_indicator, pool_adjacency = self.pool(adjacency, gcn_feature, graph_indicator)
        
        readout = torch.cat((global_avg_pool(pool, pool_graph_indicator),
                             global_max_pool(pool, pool_graph_indicator)), dim=1)
        
        fc1 = F.relu(self.fc1(readout))
        fc2 = F.relu(self.fc2(fc1))
        logits = self.fc3(fc2)
        return logits   

class modelB(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes=2):  
        super(modelB, self).__init__()
        self.gcn1 = GraphConvolution(input_dim, hidden_dim)
        self.pool1 = SelfAttentionPooling(hidden_dim, 0.5)
        self.gcn2 = GraphConvolution(hidden_dim, hidden_dim)
        self.pool2 = SelfAttentionPooling(hidden_dim, 0.5)
        self.gcn3 = GraphConvolution(hidden_dim, hidden_dim)
        self.pool3 = SelfAttentionPooling(hidden_dim, 0.5)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(), 
            nn.Linear(hidden_dim // 2, num_classes))
    
    def forward(self, adjacency, input_feature, graph_indicator):
        gcn1 = F.relu(self.gcn1(adjacency, input_feature))
        pool1, pool1_graph_indicator, pool1_adjacency = self.pool1(adjacency, gcn1, graph_indicator)
        global_pool1 = torch.cat(
            [global_avg_pool(pool1, pool1_graph_indicator),
             global_max_pool(pool1, pool1_graph_indicator)],
            dim=1)
        
        gcn2 = F.relu(self.gcn2(pool1_adjacency, pool1))
        pool2, pool2_graph_indicator, pool2_adjacency = \
            self.pool2(pool1_adjacency, gcn2, pool1_graph_indicator)
        global_pool2 = torch.cat(
            [global_avg_pool(pool2, pool2_graph_indicator),
             global_max_pool(pool2, pool2_graph_indicator)],
            dim=1)

        gcn3 = F.relu(self.gcn3(pool2_adjacency, pool2))
        pool3, pool3_graph_indicator, pool3_adjacency = \
            self.pool3(pool2_adjacency, gcn3, pool2_graph_indicator)
        global_pool3 = torch.cat(
            [global_avg_pool(pool3, pool3_graph_indicator),
             global_max_pool(pool3, pool3_graph_indicator)],
            dim=1)
        
        readout = global_pool1 + global_pool2 + global_pool3
        
        logits = self.mlp(readout)
        return logits















# %%
