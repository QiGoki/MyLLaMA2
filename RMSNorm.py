from torch import nn
import torch
"""
RMSNorm (Root Mean Square Layer Normalization)：
    2019年提出的轻量级神经网络归一化方法，
    舍弃传统LayerNorm的去均值操作，仅通过均方根对特征缩放，
    在保证模型性能的同时降低计算与存储开销，常用于大规模模型替代LayerNorm。
核心计算公式：
    RMSNorm(x) = (x / sqrt((1/n) * sum_{i=1}^n x_i^2 + ε)) * g
参数说明：
    x - 输入向量；
    n - 输入向量的维度大小；
    ε - 极小的平滑常数，避免分母为零；
    g - 可学习的缩放参数，用于调整归一化后特征的尺度。
"""

class RMSNorm(nn.Module):
    def __init__(self, dim:int, eps:float):
        super().__init__()
        # eps是为了防止除以0的情况
        self.eps = eps
        # weight是一个可学习的参数，全部初始化为1
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        # 计算RMSNorm的核心部分
        # x.pow(2).mean(-1,keepdim=True)计算了输入x的平方的均值
        # torch.rsqrt是平方根的倒数，这样就得到了RMSNorm的分母部分，再加上eps防止分母为0
        # 最后乘以x，得到RMSNorm的结果
        return x * torch.rsqrt(x.pow(2).mean(-1,keepdim=True) + self.eps)

    def forward(self,x):
        # forward函数是模型的向前传播
        # 首先将输入x转为float类型，然后进行RMSNorm，最后再转回原来的数据类型
        # 最后乘以weight，这是RMSNorm的一个可学习的缩放因子
        output = self._norm(x.float()).type_as(x)
        return output * self.weight