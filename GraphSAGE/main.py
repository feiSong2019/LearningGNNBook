'''
main.py: 基于CoraData的示例
net.py: 定义GraphSAGE模型
data.py: Cora数据集接口准备
sampling: 简单的采样接口
'''
#%%
'''实现分类
0. 超参数
1. 初始化： 模型， loss， optim
2. 训练： 数据集加载以及GPU===>输入至模型===>loss===>backward===>step
3. test
'''
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from data import CoraData
from net import GraphSage
from sampling import multihop_sampling
from collections import namedtuple

DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE=16
EROCHS=20
LEARNING_RATE=0.01

INPUT_DIM=1433
HIDDEN_DIM=[128, 7] #隐藏单元节点数目
NUM_NEIGHBORS_LIST=[10, 10] #每阶采样邻居数目
#采样邻接数需要与GCN层数保持一致
assert len(HIDDEN_DIM) == len(NUM_NEIGHBORS_LIST)
NUM_BATCH_PER_EPOCH=20 # 每个epoch循环的批次数 ?

Data=namedtuple('Data', ['x', 'y', 'adjacency_dict', 'train_mask', 'val_mask', 'test_mask'])
data=CoraData().data
x= data.x / data.x.sum(1, keepdims=True) # 归一化处理

train_index = np.where(data.train_mask)[0]
train_label = data.y[train_index]
test_index = np.where(data.test_mask)[0]
model = GraphSage(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, 
num_neighbors_list=NUM_NEIGHBORS_LIST).to(DEVICE)

criterion = nn.CrossEntropyLoss().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=5e-4)


def train():
    model.train()
    for e in range(EROCHS):
        for batch in range(NUM_BATCH_PER_EPOCH):
            batch_src_index = np.random.choice(train_index, size=(BATCH_SIZE,))
            batch_src_label = torch.from_numpy(train_label[batch_src_index]).long().to(DEVICE)
            batch_sampling_result = multihop_sampling(batch_src_index, NUM_NEIGHBORS_LIST, data.adjacency_dict)
            batch_sampling_x = [torch.from_numpy(x[idx]).float().to(DEVICE) for idx in batch_sampling_result]
            batch_train_logits = model(batch_sampling_x)
            loss = criterion(batch_train_logits, batch_src_label)
            optimizer.zero_grad()
            loss.backward()  # 反向传播计算参数的梯度
            optimizer.step()  # 使用优化方法进行梯度更新
            print("Epoch {:03d} Batch {:03d} Loss: {:.4f}".format(e, batch, loss.item()))
        test()


def test():
    model.eval()
    with torch.no_grad():
        test_sampling_result = multihop_sampling(test_index, NUM_NEIGHBORS_LIST, data.adjacency_dict)
        test_x = [torch.from_numpy(x[idx]).float().to(DEVICE) for idx in test_sampling_result]
        test_logits = model(test_x)
        test_label = torch.from_numpy(data.y[test_index]).long().to(DEVICE)
        predict_y = test_logits.max(1)[1]
        accuarcy = torch.eq(predict_y, test_label).float().mean().item()
        print("Test Accuracy: ", accuarcy)
if __name__ == '__main__':
    train()