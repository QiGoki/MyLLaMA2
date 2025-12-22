# 自用项目

本项目是基于 [Happy-LLM](https://github.com/datawhalechina/happy-llm) 教程搭建的大语言模型实现，专注于构建轻量级高效的Transformer架构模型。

## 变更记录

### 2025-12-18
- 创建项目基础结构
- 添加ModelConfig.py (50行)
  - 基于PretrainedConfig的模型超参数配置类
  - 支持Flash Attention优化(默认启用)
- 添加RMSNorm.py (37行)
  - 实现RMSNorm归一化层
  - 优化计算效率

### 2025-12-22
- 新增Attention模块
  - 实现多头注意力机制类，支持Flash Attention优化
- 新增Attention模块核心函数
  - `precompute_freqs_cis`: 预计算旋转位置嵌入的频率矩阵
  - `reshape_for_broadcast`: 调整张量形状以适应广播操作
  - `apply_rotary_emb`: 应用旋转位置嵌入到查询和键张量

## 参考
- [Happy-LLM](https://github.com/datawhalechina/happy-llm) - 从零开始的大语言模型原理与实践教程