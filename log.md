# 项目日志

## 2025-12-18 18:42:32

### 文件更新记录

1. **ModelConfig.py** (50行)
   - 基于PretrainedConfig的模型超参数配置类
   - 主要参数:
     - dim: 模型特征维度(默认768)
     - n_layers: Transformer层数(默认12)
     - n_heads: 注意力头数(默认16)
     - vocab_size: 词汇表大小(默认6144)
     - max_seq_len: 最大序列长度(默认512)
   - 支持Flash Attention优化(默认启用)

2. **RMSNorm.py** (37行)
   - 实现Root Mean Square Layer Normalization
   - 轻量级归一化方法，相比LayerNorm减少计算量
   - 核心公式: RMSNorm(x) = (x / sqrt(mean(x²) + ε)) * g
   - 包含可学习的缩放参数(weight)