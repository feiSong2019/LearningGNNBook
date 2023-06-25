'''
1. 任务： classfication based on graph
2. 简述：基于图层面的任务，需要关注整个图的全局信息----->学习得到一个优秀的全图表示向量
但参数量巨大，因此需要引入pool机制;
3. 目标： 基于自注意力机制实现池化操作
        1）通过图卷积从图中自适应地学习到节点的重要性
        2）根据节点重要度分数和拓扑结构进行池化操作，舍弃不太重要的节点， 并对邻接矩阵和节点特征进行更新         
''' 
#%%
import torch
import numpy as np
import torch.optim as optim
from net import *
from data import *
from utilits_function import *



NUM_CLASSES = 2
EPOCHS = 200    # @param {type: "integer"}
LEARNING_RATE = 0.01 # @param
WEIGHT_DECAY = 0.0001 # @param




dataset = DDDataset()
# 模型输入数据准备
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
adjacency = dataset.sparse_adjacency
normalize_adjacency = normalization(adjacency).to(DEVICE)
node_labels = tensor_from_numpy(dataset.node_labels, DEVICE)
node_features = F.one_hot(node_labels, node_labels.max().item() + 1).float()
graph_indicator = tensor_from_numpy(dataset.graph_indicator, DEVICE)
graph_labels = tensor_from_numpy(dataset.graph_labels, DEVICE)
train_index = tensor_from_numpy(dataset.train_index, DEVICE)
test_index = tensor_from_numpy(dataset.test_index, DEVICE)
train_label = tensor_from_numpy(dataset.train_label, DEVICE)
test_label = tensor_from_numpy(dataset.test_label, DEVICE)

NUM_CLASSES = 2
INPUT_DIM = node_features.size(1)
HIDDEN_DIM =    32# @param {type: "integer"}

#%%
model_g = modelA(INPUT_DIM, HIDDEN_DIM, NUM_CLASSES).to(DEVICE)
model_h = modelB(INPUT_DIM, HIDDEN_DIM, NUM_CLASSES).to(DEVICE)
model = model_g #@param ['model_g', 'model_h'] {type: 'raw'}

criterion = nn.CrossEntropyLoss().to(DEVICE)
optimizer = optim.Adam(model.parameters(), LEARNING_RATE, weight_decay=WEIGHT_DECAY)

def train():
        model.train()
        for epoch in range(EPOCHS):
                logits = model(normalize_adjacency, node_features, graph_indicator)
                loss = criterion(logits[train_index], train_label)  # 只对训练的数据计算损失值
                optimizer.zero_grad()
                loss.backward()  # 反向传播计算参数的梯度
                optimizer.step()  # 使用优化方法进行梯度更新
                train_acc = torch.eq(logits[train_index].max(1)[1], train_label).float().mean()
                print("Epoch {:03d}: Loss {:.4f}, TrainAcc {:.4}".format(epoch, loss.item(), train_acc.item()))


def test():
        model.eval()
        with torch.no_grad():
                logits = model(normalize_adjacency, node_features, graph_indicator)
                test_logits = logits[test_index]
                test_acc = torch.eq(test_logits.max(1)[1], test_label).float().mean()
        print(test_acc.item())

if __name__ == '__main__':
    train()

# %%
