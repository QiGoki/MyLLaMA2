# Tiny-K 大模型项目

本项目是基于 [Happy-LLM](https://github.com/datawhalechina/happy-llm) 教程搭建的大语言模型实现，专注于构建轻量级高效的Transformer架构模型。

## 项目概述

### 核心特性
- 实现了轻量级Transformer架构
- 支持高效的自注意力机制
- 采用RMSNorm归一化层
- 可配置的模型参数

### 技术栈
- Python 3.8+
- PyTorch 2.0+
- Transformers架构

## 文件说明

### ModelConfig.py
模型配置文件，定义了以下核心参数：
- `dim`: 模型特征维度(默认768)
- `n_layers`: Transformer层数(默认12)
- `n_heads`: 注意力头数(默认16)
- `vocab_size`: 词汇表大小(默认6144)
- `max_seq_len`: 最大序列长度(默认512)

### RMSNorm.py
实现了Root Mean Square Layer Normalization:
- 轻量级归一化方法，相比LayerNorm减少计算量
- 核心公式: RMSNorm(x) = (x / sqrt(mean(x²) + ε)) * g
- 包含可学习的缩放参数(weight)

## 变更记录

### 2025-12-18
- 创建项目基础结构
- 添加ModelConfig.py (50行)
  - 基于PretrainedConfig的模型超参数配置类
  - 支持Flash Attention优化(默认启用)
- 添加RMSNorm.py (37行)
  - 实现RMSNorm归一化层
  - 优化计算效率

## 后续计划
- 实现完整的Transformer模块
- 添加训练脚本
- 支持LoRA微调

## 参考项目
- [Happy-LLM](https://github.com/datawhalechina/happy-llm) - 从零开始的大语言模型原理与实践教程