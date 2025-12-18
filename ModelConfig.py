from transformers import PretrainedConfig

"""
ModelConfig：基于PretrainedConfig的模型超参数配置类，用于定义Tiny-K模型的核心结构参数
参数说明：
    dim: int，默认768 → 模型特征维度（如Transformer中token的embedding维度）
    n_layers: int，默认12 → Transformer的编码器/解码器堆叠层数
    n_heads: int，默认16 → 多头注意力机制的总头数
    n_kv_heads: int，默认8 → 注意力中键(Key)、值(Value)的头数（通常小于等于n_heads，用于KV缓存优化）
    vocab_size: int，默认6144 → 模型使用的词汇表大小
    hidden_dim: int，默认None → 前馈神经网络（FFN）的隐藏层维度（若为None则自动按dim计算）
    multiple_of: int，默认64 → 隐藏层维度的对齐因子（确保hidden_dim是该值的整数倍，提升硬件计算效率）
    norm_eps: float，默认1e-5 → 归一化层（如RMSNorm/LayerNorm）的平滑常数，避免分母为0
    max_seq_len: int，默认512 → 模型支持的最大序列长度（输入文本的token数量上限）
    dropout: float，默认0.0 → 正则化的dropout概率（0.0表示不使用dropout）
    flash_attn: bool，默认True → 是否启用Flash Attention（优化注意力计算的速度与显存占用）
"""


class ModelConfig(PretrainedConfig):
    model_type = "Tiny-K"

    def __init__(
            self,
            dim: int = 768,  # 模型维度
            n_layers: int = 12,  # Transformer的层数
            n_heads: int = 16,  # 注意力机制的头数
            n_kv_heads: int = 8,  # 键值头的数量
            vocab_size: int = 6144,  # 词汇表大小
            hidden_dim: int = None,  # 隐藏层维度
            multiple_of: int = 64,
            norm_eps: float = 1e-5,  # 归一化层的eps
            max_seq_len: int = 512,  # 最大序列长度
            dropout: float = 0.0,  # dropout概率
            flash_attn: bool = True,  # 是否使用Flash Attention
            **kwargs
    ):
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.multiple_of = multiple_of
        self.norm_eps = norm_eps
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.flash_attn = flash_attn
        super().__init__(**kwargs)
