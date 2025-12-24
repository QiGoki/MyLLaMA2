from typing import Optional
import torch
from torch import nn
from transformers import PreTrainedModel

from DecoderLayer import DecoderLayer
from ModelConfig import ModelConfig
from RMSNorm import RMSNorm


class Transformer(PreTrainedModel):
    config_class = ModelConfig
    last_loss: Optional[torch.Tensor]

    def __init__(self, args:ModelConfig = None):
        super().__init__(args)
        # 初始化模型参数
        self.args = args
        # 词汇表大小
        self.vocab_size = args.vocab_size
        # 层数
        self.n_layers = args.n_layers

        # 词嵌入层
        self.tok_embedding = nn.Embedding(args.vocab_size, args.dim)
        # dropout层
        self.dropout = nn.Dropout(args.dropout)
        # Decoder层
        self.layers = nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(DecoderLayer(layer_id, args))
        # 归一化层
        self.norm = RMSNorm(args.dim, eps = args.norm_eps)
        # 输出层
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

        # 将词潜入层的权重与输出层的权重共享
        self.tok_embedding.weight = self.output.weight

        # 预计算相对位置嵌入的频率































