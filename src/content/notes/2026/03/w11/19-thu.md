---
title: MiniMind — 阶段 1：环境搭建与快速体验
date: "2026-03-19"
tags: [AI LLM MiniMind]
---

# MiniMind 学习笔记 — 阶段 1：环境搭建与快速体验

> 项目地址：https://github.com/jingyaogong/minimind
> 设备环境：Apple M5 MacBook Pro / 32GB 统一内存

---

## 一、项目简介

MiniMind 是一个极简 LLM（大语言模型）训练项目，用最少的资源走通大模型训练的完整流程。

| 特性       | 说明                                                            |
|:-----------|:----------------------------------------------------------------|
| 最小参数量 | 25.8M（GPT-3 的 1/7000）                                       |
| 训练耗时   | 单张 3090 约 2 小时                                             |
| 实现方式   | PyTorch 原生实现，无第三方抽象封装                              |
| 完整流程   | Tokenizer → Pretrain → SFT → LoRA → DPO → PPO/GRPO → 模型蒸馏 |
| 模型架构   | Dense + MoE（混合专家）                                         |

---

## 二、环境搭建步骤

### 2.1 克隆项目源码

```bash
cd ~/projects
git clone https://github.com/jingyaogong/minimind.git
cd minimind
```

### 2.2 创建 conda 虚拟环境

```bash
conda create -n minimind python=3.10 -y
conda activate minimind
```

### 2.3 安装依赖

```bash
pip install -r requirements.txt
pip install transformers accelerate
```

### 2.4 下载预训练模型权重

```bash
brew install git-lfs
git lfs install
git clone https://huggingface.co/jingyaogong/MiniMind2
```

如果 HuggingFace 访问慢，可使用镜像：

```bash
git clone https://hf-mirror.com/jingyaogong/MiniMind2
```

---

## 三、运行对话测试

```bash
python eval_llm.py --load_from ./MiniMind2
```

---

## 四、两种模型加载模式

| 模式              | 命令                                         | 权重路径                  | 适用场景               |
|:------------------|:---------------------------------------------|:--------------------------|:-----------------------|
| 原生 PyTorch      | python eval_llm.py                           | ./out/full_sft_512.pth    | 自己从零训练的模型     |
| Transformers 格式 | python eval_llm.py --load_from ./MiniMind2   | ./MiniMind2/              | HuggingFace 下载的模型 |

**代码 vs 权重：**

| 概念 | 来源            | 内容                             |
|:-----|:----------------|:---------------------------------|
| 代码 | minimind 仓库   | 模型的结构定义、训练逻辑、评估脚本 |
| 权重 | MiniMind2 仓库  | 训练好的参数数值                 |

---

## 五、项目目录结构（初步认识）

```
minimind/
├── minimind/            # 模型核心代码
│   └── model/           # 模型结构定义（Dense + MoE）
├── dataset/             # 数据集目录
├── MiniMind2/           # 下载的预训练权重（Transformers 格式）
├── out/                 # 自己训练产出的权重（目前为空）
├── train_pretrain.py    # 预训练脚本
├── train_full_sft.py    # 全参数 SFT 脚本
├── train_lora.py        # LoRA 微调脚本
├── eval_llm.py          # 模型评估/对话测试
└── requirements.txt     # 依赖清单
```

---

## 六、关键概念速查

| 概念              | 说明                                                             |
|:------------------|:-----------------------------------------------------------------|
| LLM               | Large Language Model，大语言模型                                 |
| PyTorch           | 最流行的深度学习框架，MiniMind 的基础                            |
| transformers      | HuggingFace 开发的模型工具库，几乎所有开源大模型通过它发布       |
| conda             | Python 环境管理工具，可创建隔离的虚拟环境                        |
| pip               | Python 包安装工具                                                |
| git-lfs           | Git Large File Storage，用于下载大文件（如模型权重）             |
| HuggingFace       | AI 模型社区和工具链，相当于 AI 界的 GitHub                       |
| .pth 文件         | PyTorch 原生权重格式                                             |
| Transformers 格式 | HuggingFace 标准格式，含 config.json、权重文件、tokenizer        |
| MPS               | Metal Performance Shaders，Apple 芯片的 GPU 加速后端             |

---

## 七、踩坑记录

| 编号 | 报错信息                                    | 原因                             | 解决方案                              |
|:-----|:--------------------------------------------|:---------------------------------|:--------------------------------------|
| 1    | can't open file 'eval_model.py'             | 文件名记错                       | 正确文件名是 eval_llm.py             |
| 2    | FileNotFoundError: ./out/full_sft_512.pth   | 默认加载原生 PyTorch 权重        | 加参数 --load_from ./MiniMind2       |
| 3    | No module named 'transformers'              | 未安装 transformers 库           | pip install transformers accelerate   |

---

## 八、下一步计划

**阶段 2：理解模型架构** — Transformer Decoder-Only 架构，RMSNorm、RoPE、SwiGLU 等核心组件，结合 MiniMind 源码逐层拆解。

# 补充提问
1. 完整流程	Tokenizer → Pretrain → SFT → LoRA → DPO → PPO/GRPO → 模型蒸馏
2. 模型架构	Dense + MoE（混合专家）

## 1. 完整训练流程

把它想象成**培养一个小孩**：

| 步骤 | 做什么 | 大白话 |
|:-----|:------|:------|
| **Tokenizer** | 造字典 | 先规定"哪些字/词是我认识的"，把文字变成数字 |
| **Pretrain** | 海量阅读 | 读几百本书，学会基本语感（能说人话，但不会对话） |
| **SFT** | 上课学习 | 用"老师问 → 标准答案"的方式，教会它对话 |
| **LoRA** | 专项补习 | 只调一小部分参数，快速学会某个特定技能（比如医疗问答） |
| **DPO** | 纠正三观 | 给两个回答，告诉它"A 比 B 好"，学会判断好坏 |
| **PPO/GRPO** | 深度纠正 | 更复杂的纠正方式，通过打分+奖惩让回答越来越好 |
| **模型蒸馏** | 向学霸抄笔记 | 大模型当老师，小模型学着模仿老师的输出 |

> 一句话总结：**先学认字 → 再学说话 → 再学对话 → 再学做人 → 最后向高手偷师**

---

## 2. Dense 和 MoE 两种架构

| 架构 | 大白话 | 特点 |
|:-----|:------|:-----|
| **Dense（密集）** | 每个问题都让**全部神经元**一起干活 | 简单直接，所有参数每次都参与计算 |
| **MoE（混合专家）** | 养一组"专家"，每个问题只挑**几个专家**来回答 | 参数总量大，但每次只用一部分，所以速度快 |

打个比方：

- **Dense** = 一个全科医生，什么病都自己看
- **MoE** = 一个医院，有内科、外科、眼科等专家，来了病人先分诊，再派对应专家处理

> MoE 的好处：模型可以很大（知识多），但推理时只激活一小部分（速度快）。DeepSeek 用的就是 MoE 架构。

---
