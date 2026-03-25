---
title: MiniMind — 学习进度总结
date: "2026-03-25"
tags: [AI, MiniMind]
---

# 📋 MiniMind 学习进度总结

## 学习者背景
- Apple M5 MacBook Pro（32GB），无 NVIDIA GPU，使用 PyTorch MPS 后端
- Python 初学，PyTorch 零经验，深度学习零基础
- 目标：理解原理、二次开发、求职

---

## 已完成：阶段 1 —— 环境搭建与快速体验
- 克隆项目、创建虚拟环境、安装依赖
- 下载预训练权重，运行 `eval_llm.py` 对话测试
- 确认 MPS 后端可用

---

## 已完成：阶段 2（进行中）—— 模型架构

### 2.1 Transformer 整体架构
- MiniMind 采用 **Decoder-Only** 架构（同 GPT、LLaMA）
- 核心流程：Embedding → N 层 Decoder Block → RMSNorm → Linear 输出
- 每个 Decoder Block：RMSNorm → Attention → 残差连接 → RMSNorm → FFN → 残差连接
- MiniMind 默认配置：`hidden_size=512, num_heads=8, num_kv_heads=2, num_layers=8`

### 2.2 RMSNorm
- 只用均方根做归一化，省去均值计算，比 LayerNorm 更快
- 公式：`x_norm = x / RMS(x) * γ`，其中 `RMS(x) = sqrt(mean(x²) + ε)`
- γ 是可学习的缩放参数，ε 防止除零

### 2.3 RoPE（旋转位置编码）
- 通过旋转向量的角度来编码位置信息
- 核心性质：两个 Token 的点积只取决于它们的**相对距离**
- 把向量两两分组，每组在二维平面上按位置旋转不同角度
- 低维组旋转快（捕捉局部关系），高维组旋转慢（捕捉远距离关系）
- 用复数乘法高效实现旋转，直接作用在 Q 和 K 上

### 2.4 SwiGLU 激活函数（FFN）
- FFN 是 Attention 之后的逐位置特征变换，负责"深度加工"信息
- SwiGLU = Swish 激活 + 门控机制（GLU），两条路径相乘
- 三个线性层：gate 路径（Swish 激活）× up 路径，再经 down 投影回原始维度
- 隐藏层维度通常是输入维度的数倍（MiniMind 中由 `intermediate_size` 控制）

### 2.5 Self-Attention
- 每个 Token 通过 Q/K/V 三个角色与其他 Token 交互
- Q = "我在找什么"，K = "我能提供什么"，V = "我的实际内容"
- 公式：`Attention(Q,K,V) = softmax(Q·K^T / √d_k) · V`
- 除以 √d_k 防止点积过大导致 softmax 饱和、梯度消失

### 2.6 Causal Mask
- 自回归生成要求每个 Token 只能看到它之前的 Token，不能偷看未来
- 用上三角矩阵将未来位置的注意力分数设为 -∞，softmax 后变为 0
- MiniMind 实现：`torch.triu(full(-inf), diagonal=1)` 加到 scores 上

---

## ⏭️ 下一步应继续学习
1. Multi-Head Attention → 
2. GQA → 
3. KV-Cache → 
4. MoE →
5. 更多...
