# 自用项目

本项目是基于 [Happy-LLM](https://github.com/datawhalechina/happy-llm) 教程搭建的大语言模型实现。

## 变更记录

### 2025-12-18
- 创建项目基础结构
- 添加ModelConfig.py (50行)
  - 基于PretrainedConfig的模型超参数配置类
  - 支持Flash Attention优化(默认启用)
- 添加RMSNorm.py (37行)
  - 实现RMSNorm归一化层
  - 优化计算效率

## 参考
- [Happy-LLM](https://github.com/datawhalechina/happy-llm) - 从零开始的大语言模型原理与实践教程