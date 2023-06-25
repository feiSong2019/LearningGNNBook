import torch
import numpy as np
import scipy.sparse as sp

def tensor_from_numpy(x, device):
    return torch.from_numpy(x).to(device)


def normalization(adjacency):
    """计算 L=D^-0.5 * (A+I) * D^-0.5,

    Args:
        adjacency: sp.csr_matrix.

    Returns:
        归一化后的邻接矩阵，类型为 torch.sparse.FloatTensor
    """
    adjacency += sp.eye(adjacency.shape[0])    # 增加自连接
    degree = np.array(adjacency.sum(1))
    d_hat = sp.diags(np.power(degree, -0.5).flatten())
    L = d_hat.dot(adjacency).dot(d_hat).tocoo()
    # 转换为 torch.sparse.FloatTensor
    indices = torch.from_numpy(np.asarray([L.row, L.col])).long()
    values = torch.from_numpy(L.data.astype(np.float32))
    tensor_adjacency = torch.sparse.FloatTensor(indices, values, L.shape)
    return tensor_adjacency