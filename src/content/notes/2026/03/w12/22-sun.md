---
title: MiniMind — 阶段 2.2：RMSNorm
date: "2026-03-22"
tags: [AI,MiniMind]
---

# mean(x) 和 std(x) 是什么？

### ① mean(x)：均值（Mean）

就是**求平均**：

```
x = [150.0, 200.0, 180.0]

mean(x) = (150 + 200 + 180) / 3 = 176.7
```

### ② std(x)：标准差（Standard Deviation）

衡量数据**离均值有多远**，也就是"数据有多分散"：

```
x = [150.0, 200.0, 180.0]，mean = 176.7

① 每个值减去均值，再平方：
   (150 - 176.7)² = 712.9
   (200 - 176.7)² = 542.9
   (180 - 176.7)² = 10.9

② 求平均：
   (712.9 + 542.9 + 10.9) / 3 = 422.2

③ 开根号：
   sqrt(422.2) ≈ 20.5
```

所以 `std(x) ≈ 20.5`，意思是这组数据平均偏离均值约 20.5。

---

# LayerNorm 完整过程

```
x = [150.0, 200.0, 180.0]
mean = 176.7, std = 20.5

LayerNorm(x) = (x - mean) / std * γ + β

① 减均值（centering，中心化）：
   [150-176.7, 200-176.7, 180-176.7] = [-26.7, 23.3, 3.3]

② 除以标准差（scaling，缩放）：
   [-26.7/20.5, 23.3/20.5, 3.3/20.5] = [-1.30, 1.14, 0.16]

③ 乘 γ，加 β（都是 learnable parameters）
```

---

# LayerNorm和RMSNorm的差异

现在回到刚才的问题，其实 RMSNorm 比 LayerNorm **少了两个操作**：

| 操作 | LayerNorm | RMSNorm |
|------|-----------|---------|
| **减均值（centering）** | ✅ `x - mean(x)` | ❌ 没有 |
| 缩放 | ✅ `/ std(x)` | ✅ `/ RMS(x)` |
| 乘 γ | ✅ | ✅ |
| **加 β（bias）** | ✅ `+ β` | ❌ 没有 |

你发现的 `+β` 是第一个少的 ✅

还多少了一步：**不减均值（no centering）**✅

---

# 为什么 MiniMind（和 LLaMA）选择 RMSNorm？

一句话：**效果几乎一样，但计算更快**。

- 少算 mean → 省计算
- 少了 β 参数 → 省内存
- 实验证明对 LLM 来说，centering 对效果影响很小

---

### 🤔 思考题

现在我们理解了 RMSNorm 的数学原理。接下来我想让你看看**真实代码**长什么样。

在你之前克隆的 MiniMind 项目中，打开这个文件：

```
minimind/model/model.py
```

找到 `class RMSNorm`，看看它的 `forward` 方法。

问题：代码里有一个很小的数 `self.eps`（即公式里的 ε），你觉得它是干什么用的？为什么需要它？

（提示：想想如果 RMS(x) 的值等于 0 会发生什么）

### 回答
没有 `model.py` 文件，实际项目里是 `model_minimind.py` 文件：

```python
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self.weight * self._norm(x.float()).type_as(x)
```

`ε` 可能是最初输入的值，避免多次转换之后与原始输入偏差过大？

为`0`会导致计算异常？根据公式来看，`RMS(x)` 会作为除数，而`0`是不能作为除数的

---

# ε 的作用：防止除以零（Division by Zero）
你的第二个猜测**完全正确！** ✅

```
RMS(x) = sqrt( mean(x²) + ε )
```

如果所有值恰好为 0：
```
x = [0.0, 0.0, 0.0]
mean(x²) = 0

没有 ε → sqrt(0) = 0 → x / 0 = 💥 报错！
有了 ε → sqrt(0 + 0.00001) = 0.00316 → 安全！
```

`ε = 1e-5` 即 `0.00001`，极小，不影响正常计算，但能**保底**。

这是深度学习中非常常见的技巧，叫做 **numerical stability trick**。

---

# 逐行读懂 RMSNorm 代码
### 第一部分：`__init__`（初始化）

```python
def __init__(self, dim: int, eps: float = 1e-5):
    super().__init__()
    self.eps = eps                                  # 存储 ε
    self.weight = nn.Parameter(torch.ones(dim))     # 就是公式里的 γ
```

- `dim`：向量的维度（比如 512）
- `nn.Parameter`：告诉 PyTorch "这个变量需要训练"
- `torch.ones(dim)`：初始值全是 1，即 `γ = [1, 1, 1, ..., 1]`

### 第二部分：`_norm`（核心计算）

```python
def _norm(self, x):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
```

拆解每一步：

```python
x.pow(2)              # x²，每个元素求平方
  .mean(-1,           # 沿最后一个维度求均值（mean）
        keepdim=True) #   保持维度不变（方便后续运算）
  + self.eps          # 加上 ε，防止为 0

torch.rsqrt(...)      # rsqrt = 1/sqrt(...)，即"先开根号，再取倒数"

x * ...               # x 乘以 1/sqrt(...)，等价于 x / sqrt(...)
```

对应公式：`x / sqrt(mean(x²) + ε)` ✅

> 💡 为什么用 `rsqrt`（倒数平方根）而不是 `x / sqrt(...)`？
> 因为乘法比除法**计算更快**，在 GPU 上尤其如此。

### 第三部分：`forward`（前向传播）

```python
def forward(self, x):
    return self.weight * self._norm(x.float()).type_as(x)
```

```python
x.float()          # 转为 float32 高精度，确保计算精确
self._norm(...)     # 做归一化
.type_as(x)         # 转回原来的数据类型（可能是 float16/bfloat16）
self.weight * ...   # 乘以 γ（learnable parameter）
```

> 💡 为什么要先转 float32 再转回来？
> 模型训练常用半精度（float16/bfloat16）来省内存加速，
> 但归一化涉及求平方和开根号，精度不够容易出错，
> 所以**临时**用 float32 算，算完再转回去。这叫 **mixed precision（混合精度）**。

---

## 🔨 动手任务

在终端中运行以下 Python 代码，亲手验证 RMSNorm 的计算：

```python
import torch
import torch.nn as nn

# 创建一个 RMSNorm（维度=3）
rms = nn.Module()  # 我们手动算

x = torch.tensor([[150.0, 200.0, 180.0]])  # 1个Token，3维

# 手动计算 RMSNorm
eps = 1e-5
rms_val = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
output = x * rms_val

print(f"输入: {x}")
print(f"RMS 倒数: {rms_val}")
print(f"输出: {output}")
```

看看输出是否接近我们之前手算的 `[0.84, 1.12, 1.01]`。

---

## 🤔 思考题

RMSNorm 在 MiniMind 模型中被用在了**哪些位置**？

提示：在 `model_minimind.py` 中搜索 `RMSNorm`，看看它出现在哪里。想想为什么要在这些位置做归一化？

---

**下一步预告**：零件 ② —— **RoPE（Rotary Position Embedding，旋转位置编码）**，让模型知道每个 Token 的"位置信息"。

## 回答
动手题运行结果：`输出: tensor([[0.8434, 1.1245, 1.0120]])`

思考题：

1. 在 `MiniMindBlock` 的 `__init` 中有2处
```python
class MiniMindBlock(nn.Module):
    def __init__(self, layer_id: int, config: MiniMindConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.self_attn = Attention(config)

        self.layer_id = layer_id
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    # ...其他代码
```

这里之前有提到过，正是 `TransformBlock`的内部转换，整个流程是：
1. 输入
2. RMSNorm
3. Attention
4. 残差连接
5. RMSNorm
6. FFN / SwiGLu
7. 残差连接
8. 输出

所以我理解是在输入之后 和 Attention输出之后分别进行了一次 RMSNorm

9.  在 `MiniMindModel` 的 `__init` 中有1处
```python
class MiniMindModel(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([MiniMindBlock(l, config) for l in range(self.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        freqs_cos, freqs_sin = precompute_freqs_cis(dim=config.hidden_size // config.num_attention_heads,
                                                    end=config.max_position_embeddings, rope_base=config.rope_theta,
                                                    rope_scaling=config.rope_scaling)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    # ...其他代码
```

这里我理解就是整个初始化的时候进行了一次RMSNorm，这里从代码看应该是完成 MiniMindBlock 的多层处理后，对最终的一个结果进行RMSNorm

# RMSNorm在哪里使用？
三处全部找到，理解也非常到位！👏

---

## 为什么在这三个位置做 RMSNorm？

### 位置 ①② ：MiniMindBlock 内部（每一层都有）

你总结的流程完全正确。我补充一个重要术语：

#### Pre-Norm
这种**先归一化，再计算**的模式叫 **Pre-Norm**：

```
Pre-Norm（MiniMind / LLaMA 的做法）:
   → RMSNorm → Attention → 残差连接
   → RMSNorm → FFN      → 残差连接
```

#### Post-Norm
还有一种相反的做法叫 **Post-Norm**（原始 Transformer 论文的做法）：

```
Post-Norm（原始 Transformer）:
   → Attention → 残差连接 → LayerNorm
   → FFN      → 残差连接 → LayerNorm
```

为什么现在主流 LLM 都用 Pre-Norm？

| | Pre-Norm | Post-Norm |
|--|----------|-----------|
| 归一化时机 | 计算**之前** | 计算**之后** |
| 训练稳定性 | ✅ 更稳定 | ⚠️ 深层网络容易不稳定 |
| 是否需要 warm-up | 不太依赖 | 强烈依赖 learning rate warm-up |
| 使用者 | LLaMA, GPT-3, MiniMind | 原始 Transformer, BERT |

> 💡 **直觉理解**：
> Pre-Norm 就像"先洗菜再炒"——保证每次 Attention / FFN 拿到的输入都是干净的。
> Post-Norm 像"先炒完再洗"——如果中间爆了（数值爆炸），来不及补救。

### 位置 ③：所有层结束后的最终 RMSNorm

你的理解完全正确。数据经过多层 MiniMindBlock 后，数值可能又漂移了。

在送入最后的 **output head**（输出层，预测下一个 Token）之前，再做一次归一化，确保稳定：

```
Embedding → [Block 0 → Block 1 → ... → Block N] → RMSNorm → Output Head
                                                     ↑
                                                  你找到的位置 ③
```

---

## 完整的 RMSNorm 地图

```
输入 Token IDs
     ↓
 Embedding（查表，得到向量）
     ↓
┌─── Block 0 ──────────────────┐
│  ① RMSNorm → Attention → 残差 │
│  ② RMSNorm → FFN       → 残差 │
└──────────────────────────────┘
     ↓
┌─── Block 1 ──────────────────┐
│  ① RMSNorm → Attention → 残差 │
│  ② RMSNorm → FFN       → 残差 │
└──────────────────────────────┘
     ↓
    ...（重复 N 层）
     ↓
 ③ RMSNorm（最终归一化）
     ↓
 Output Head（预测下一个 Token）
```

---

## 本节要点总结

| 要点 | 内容 |
|------|------|
| Pre-Norm vs Post-Norm | MiniMind 用 Pre-Norm，训练更稳定 |
| RMSNorm 出现 3 处 | Attention 前、FFN 前、所有层之后 |
| 目的一致 | 每次重要计算前/后，稳定数值范围 |

---

## 🤔 思考题（承上启下）

我们现在知道了 Transformer Block 内部的流程：

```
→ RMSNorm → Attention → 残差 → RMSNorm → FFN → 残差
```

其中 **Attention（注意力机制）** 是最核心的部分，它让每个 Token 能"看到"其他 Token。

但有一个问题：Attention 计算的是 Token 之间的**关系强弱**，它本身并不知道 Token 的**顺序**。

比如这两句话：

```
"猫 吃 鱼"
"鱼 吃 猫"
```

如果没有位置信息，Attention 会认为它们完全一样（因为包含的 Token 相同）。

那么问题来了：**怎么让模型知道每个 Token 在句子中的位置？**

这就是下一个零件要解决的问题。你有什么猜想吗？或者准备好了，我们就进入零件 ② ——**RoPE（Rotary Position Embedding，旋转位置编码）** 🚀