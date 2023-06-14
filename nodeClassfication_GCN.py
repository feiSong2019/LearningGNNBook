'''
realize node classfication based on GCN network
1. setup
2. prepare data
3. define GCN layer
4. define model (Net)
5. train model
'''
# Cora datasets: 2708 papers + 5429 edges---->papers are devided into 7 classfications. Each paper is featured with one dimension with 1433
#%%
import itertools                    # 迭代器
import os                           # 对文件和文件夹进行操作
import os.path as osp               # 文件、文件夹路径
import pickle                       # 将python文件、对象进行二值化，以实现python对象的存储和恢复
import urllib                       # HTTP请求库
from collections import namedtuple  # collections 为解决特定问题而设计一些容器: https://blog.csdn.net/weixin_41261833/article/details/118668189
import ssl                          # ssl 提供网络连接库
ssl._create_default_https_context = ssl._create_unverified_context

import numpy as np
import scipy.sparse as sp           # 处理稀疏矩阵
import torch
import torch.nn as nn               # torch.nn 与 torch.nn.functional之间的区别： https://blog.csdn.net/orangerfun/article/details/122667273
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
import matplotlib.pyplot as plt
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')





#%%
# data preprocess: download-->normalize-->stroge
# x: node fearute [2078*1433] ; y: labels [7]; adjacency: relationship [2078*2078] --- scipy.sparse.coo_matrix;
# train_mask, val_mask, test_mask: for partitioning train test validation
def tensor_from_numpy(x, device):
    return torch.from_numpy(x).to(device)

Data=namedtuple('Data', ['x' , 'y' , 'adjacency' , 'train_mask' , 'val_mask' , 'test_mask']) # namedtuple user: https://blog.csdn.net/m0_56312629/article/details/124182297

# s1.1 define Coradata class
class Coradata(object):
    download_url="https://github.com/kimiyoung/planetoid/raw/master/data"
    filenames=["ind.cora.{}".format(name) for name in ['x', 'tx', 'allx', 'y', 'ty', 'ally', 'graph', 'test.index']]
    def __init__(self, data_root='cora', rebuild=False):
        ''' A. 函数功能： data download, process, loading
        处理之后的数据集可以通过属性.data获得，它将返回一个数据对象，包括如下几个部分：
        x：节点的特征，2708*1433，类型为 np.ndarry
        y: 节点的标签，7， 类型为 np.ndarray
        adjacency: 邻接矩阵，2708*2708， 类型为 scipy.sparse.coo_matrix
        train_mask: 训练集掩码向量，维度为2708， 当节点属于训练集时，相应位置为True，否则为False
        val_mask: 验证集掩码
        test_mask: 测试集掩码
        '''

        ''' B. 输入参数
            data_root: string, optional
                       存放数据的目录，原始数据路径：{data_root}/raw
                       缓存数据路径： {data_root}/processed_cora.pkl
            rebuild: boolean, optional
                       是否需要重新构建数据集，当设置为True时，如果缓存数据也会重建数据
        '''
        self.data_root= data_root
        save_file=osp.join(self.data_root, "processed_cora.pkl")
        if osp.exists(save_file) and not rebuild:
            print("Using Cached file: {}".format(save_file))
            self._data=pickle.load(open(save_file,'rb'))
        else:
            self.maybe_download()
            self._data=self.process_data()
            with open(save_file, 'wb') as f:
                pickle.dump(self.data,f)
            print('Cached file: {}'.format(save_file))
    @property
    def data(self):
        '''返回数据对象，包括： x, y, adjacency, train_mask, val_mask, test_mask'''
        return self._data
    
    def process_data(self):
        '''处理数据，得到节点特征，邻接矩阵，验证集，测试集，训练集'''
        print('Process data ....')
        _, tx, allx, y, ty, ally, graph, test_index = [self.read_data(osp.join(self.data_root, 'raw', name)) for name in self.filenames]
        train_index = np.arange(y.shape[0])
        val_index = np.arange(y.shape[0], y.shape[0] + 500)
        sorted_test_index = sorted(test_index)

        x = np.concatenate((allx, tx), axis=0)
        y = np.concatenate((ally, ty), axis=0).argmax(axis=1)

        x[test_index]=x[sorted_test_index]
        y[test_index]=y[sorted_test_index]
        num_nodes = x.shape[0]

        train_mask=np.zeros(num_nodes, dtype=np.bool)
        val_mask=np.zeros(num_nodes, dtype=np.bool)
        test_mask=np.zeros(num_nodes, dtype=np.bool)
        train_mask[train_index]=True
        val_mask[val_index]=True
        test_mask[test_index]=True
        adjacency=self.build_adjacency(graph)
        print("Node's feature shape: ", x.shape)
        print("Node's label shape: ", y.shape)
        print("Adjacency's shape ", adjacency.shape)
        print("Number of training nodes: ", train_mask.sum())
        print("Number of validation nodes: ", val_mask.sum())
        print("Number of test nodes: ", test_mask.sum())

        return Data(x=x, y=y, adjacency=adjacency, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

    def maybe_download(self):
        save_path = os.path.join(self.data_root, "raw")
        for name in self.filenames:
            if not osp.exists(osp.join(save_path, name)):
                self.download_data('{}/{}'.format(self.download_url, name), save_path) 
    
    @staticmethod
    def build_adjacency(adj_dict):
        '''根据邻接表创建邻接矩阵'''
        edge_index=[]
        num_nodes=len(adj_dict)
        for src,dst in adj_dict.items():
            edge_index.extend([src, v] for v in dst)
            edge_index.extend([v, src] for v in dst)
        edge_index=list(k for k,_ in itertools.groupby(sorted(edge_index))) # 去除重复的边
        edge_index=np.asarray(edge_index)
        adjacency=sp.coo_matrix((np.ones(len(edge_index)),(edge_index[:,0],edge_index[:,1])),shape=(num_nodes,num_nodes),dtype="float32")
        return adjacency
    
    @staticmethod
    def read_data(path):
        """使用不同的方式读取原始数据以进一步处理"""
        name = osp.basename(path)
        if name == "ind.cora.test.index":
            out = np.genfromtxt(path, dtype="int64")
            return out
        else:
            out = pickle.load(open(path, "rb"), encoding="latin1")
            out = out.toarray() if hasattr(out, "toarray") else out
            return out
    
    @staticmethod
    def download_data(url, save_path):
        """数据下载工具，当原始数据不存在时将会进行下载"""
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        data = urllib.request.urlopen(url)
        filename = os.path.split(url)[-1]

        with open(os.path.join(save_path, filename), 'wb') as f:
            f.write(data.read())

        return True
    
    @staticmethod
    def normalization(adjacency):
        """计算 L=D^-0.5 * (A+I) * D^-0.5"""
        adjacency += sp.eye(adjacency.shape[0])    # 增加自连接
        degree = np.array(adjacency.sum(1))
        d_hat = sp.diags(np.power(degree, -0.5).flatten())
        return d_hat.dot(adjacency).dot(d_hat).tocoo()

# %%
# s2: define GCN layer
class GraphConvolution(nn.Module):
    # 1: init: 初始化forward函数需要用到的变量；weight--->input_dim & output_dim
    # 2: 对参数进行重置;
    # 3: forward 堆叠模型: LXW(W为模型参数)
    def __init__(self, input_dim, output_dim, use_bias=True):
        super(GraphConvolution, self).__init__()
        self.input_dim=input_dim
        self.output_dim=output_dim
        self.weight=nn.Parameter(torch.Tensor(input_dim, output_dim))
        self.use_bias=use_bias
        if self.use_bias:
            self.bias=nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    def reset_parameters(self):
        #init.kaming_uniform_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias)
    def forward(self, adjacency, input_feature):
        support=torch.mm(input_feature, self.weight)
        output=torch.sparse.mm(adjacency, support)
        if self.use_bias:
            out += self.bias
        return output
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.input_dim) + ' -> ' \
            + str(self.output_dim) + ')'

# %%
# s3: define two GCN layer
# input:1433;   hidden:16;  output:7-----类别数目; 激活函数ReLU
class GcnNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, use_bias=True):
        super(GcnNet,self).__init__()
        self.input_dim=input_dim
        self.hidden_dim=hidden_dim
        self.output_dim=output_dim
        self.gcn1=GraphConvolution(self.input_dim, self.hidden_dim)
        self.gcn2=GraphConvolution(self.hidden_dim, self.output_dim)

    def reset_parameter(self):
        self.gcn1.reset_parameter()
        self.gcn2.reset_parameter()
    
    def forward(self, adjacency, feature):
        h=F.relu(self.gcn1(adjacency, feature))
        logits = self.gcn2(adjacency, h)
        return logits

# s4: train
''' 
 s4.1  定义超参数: 学习率， 下降精度， epoch
 s4.2  加载数据集，并转为tensor向量
 s4.3  实例化对象： 模型， loss， optimizer
 s4.4  定义训练函数： 前向传播，loss计算， 方向传播， 梯度更新
       每个epoch结束后进行： loss计算，训练精度，验证集精度
'''
LEARNING_RATE=0.1
WEIGHT_DACAY=5E-4
EPOCHS=200
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'

# x, y, adjacency, train_mask, val_mask, test_mask
dataset=Coradata().data
nodefeature=dataset.x/dataset.x.sum(1,keepdims=True)
tensor_x=tensor_from_numpy(nodefeature, DEVICE)
tensor_y=tensor_from_numpy(dataset.y, DEVICE)
tensor_train_mask=tensor_from_numpy(dataset.train_mask, DEVICE)
tensor_val_mask=tensor_from_numpy(dataset.val_mask, DEVICE)
tensor_test_mask=tensor_from_numpy(dataset.test_mask, DEVICE)
normalize_adjacency=Coradata.normalization(dataset.adjacency)

num_nodes, input_dim=nodefeature.shape
indices=torch.from_numpy(np.asarray([normalize_adjacency.row, normalize_adjacency.col]).astype('int64')).long()

values = torch.from_numpy(normalize_adjacency.data.astype(np.float32))
tensor_adjacency=torch.sparse.FloatTensor(indices, values, (num_nodes,num_nodes)).to(DEVICE)

#%%
hidden_dim=16
output_dim=7
model=GcnNet(input_dim, hidden_dim, output_dim).to(DEVICE)
criterion=nn.CrossEntropyLoss().to(DEVICE)
optimizer=optim.Adam(model.parameters(),
                    lr=LEARNING_RATE,
                    weight_decay=WEIGHT_DACAY)

def train():
    loss_history=[]
    val_acc_history=[]
    model.train() 
    train_y = tensor_y[tensor_train_mask]
    for epoch in range(EPOCHS):
        logits= model(tensor_adjacency, tensor_x)
        train_mask_logits=logits[tensor_train_mask]
        loss=criterion(train_mask_logits, train_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_acc,_,_=test(tensor_train_mask)
        val_acc,_,_=test(tensor_val_mask)
        loss_history.append(loss.item())
        val_acc_history.append(val_acc.item())
        print("Epoch {:03d}: Loss {:.4f}, TrainAcc {:.4}, ValAcc {:.4f}".format(
            epoch, loss.item(), train_acc.item(), val_acc.item()))
    return loss_history, val_acc_history    

def test(mask):
    model.eval()
    with torch.no_grad():
        logits = model(tensor_adjacency, tensor_x)
        test_mask_logits = logits[mask]
        predict_y = test_mask_logits.max(1)[1]
        accuarcy = torch.eq(predict_y, tensor_y[mask]).float().mean()
    return accuarcy, test_mask_logits.cpu().numpy(), tensor_y[mask].cpu().numpy()    

def plot_loss_with_acc(loss_history, val_acc_history):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(range(len(loss_history)), loss_history,
             c=np.array([255, 71, 90]) / 255.)
    plt.ylabel('Loss')
    
    ax2 = fig.add_subplot(111, sharex=ax1, frameon=False)
    ax2.plot(range(len(val_acc_history)), val_acc_history,
             c=np.array([79, 179, 255]) / 255.)
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    plt.ylabel('ValAcc')
    
    plt.xlabel('Epoch')
    plt.title('Training Loss & Validation Accuracy')
    plt.show()

loss, val_acc = train()
test_acc, test_logits, test_label = test(tensor_test_mask)
print("Test accuarcy: ", test_acc.item())

plot_loss_with_acc(loss, val_acc)
# %%
