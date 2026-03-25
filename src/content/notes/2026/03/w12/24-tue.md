---
title: MiniMind — 阶段 2.5：Attention、Causal Mask
date: "2026-03-24"
tags: [AI, MiniMind]
---

两个回答都非常好！👏 精准度超出零基础水平。让我补充完善一下。

# 1. FFN

## FFN vs Attention 分别做了什么

| | 你的回答 | 更精确的说法 |
|--|---------|-------------|
| FFN | 深度加工信息，得到更本质的语义 | ✅ **对每个 Token 独立地深加工**（存储和调用知识） |
| Attention | 计算不同 Token 之间的关联程度 | ✅ **让 Token 之间互相交流信息** |

> FFN核心公式：output = self.w2(F.silu(self.w1(x)) * self.w3(x))

Transformer 一层的完整流程：

```
输入 Token 序列
      │
      ▼
  ┌─────────┐
  │Attention │ ← 开会：Token 之间互相交流
  └────┬─────┘
       │
      ▼
  ┌─────────┐
  │   FFN   │ ← 回工位：每个 Token 独立深加工
  └────┬─────┘
       │
      ▼
  输出（作为下一层的输入）

这个过程重复 8 次（MiniMind 有 8 层）

就像：开会 → 干活 → 开会 → 干活 → ... 重复 8 轮
```

## FFN和SwiGLU的关系
**FFN（Feed-Forward Network） 是结构，SwiGLU 是该结构的一种具体实现方案。**

一句话：**FFN 是 Transformer Block 中的一个功能模块，SwiGLU 是这个模块的具体实现方式之一。**

# 2. Attention
## 为什么需要 Attention

FFN 对每个 token 独立处理，即 token 之间没有任何信息交换

语言需要上下文。比如：

```
"它 太 大 了"
"它"指什么？必须看前文其他 token 才能知道。FFN 做不到这一点。
```

**Attention 的作用就是让每个 token 能看到序列中其他 token 的信息，建立 token 间的依赖关系。**

## Attention 的 Q、K、V 是什么

```
每个 Token 会生成三个向量：

Q（Query）  = "我在找什么？"
K（Key）    = "我能提供什么？"
V（Value）  = "我的具体内容是什么？"
```

比如：

```
"小明 把 苹果 给了 小红 ， 她 很 开心"

当模型处理"她"这个 Token 时：

"她"生成 Q："我在找——我指代的是谁？"

每个词都有 K：
  "小明"的 K："我是——男性人名"      → 匹配度：低 (0.05)
  "苹果"的 K："我是——水果"          → 匹配度：极低 (0.01)
  "小红"的 K："我是——女性人名"      → 匹配度：高 (0.80)
  "开心"的 K："我是——情绪词"        → 匹配度：中 (0.14)

然后按匹配度加权获取 V：
  "她"的最终理解 = 0.05×小明的V + 0.01×苹果的V + 0.80×小红的V + 0.14×开心的V
                   ≈ 主要是"小红"的信息
```

用数学表达：

```
Attention(Q, K, V) = softmax(Q × K^T / √d) × V
                     ↑                         ↑
                  注意力分数                 加权求和
                 （谁和谁相关）          （提取相关内容）
```

### ✅ 动手实操

用代码感受 Q、K 匹配的过程：

```python
import torch
import torch.nn.functional as F

# 假设 3 个 Token，每个用 4 维向量表示
# Token 0: "小红"  Token 1: "很"  Token 2: "她"

Q = torch.tensor([
    [0.1, 0.2, 0.3, 0.1],   # "小红"的 Q
    [0.0, 0.1, 0.0, 0.2],   # "很"的 Q
    [0.9, 0.1, 0.8, 0.0],   # "她"的 Q ← 重点观察这个
])

K = torch.tensor([
    [0.8, 0.2, 0.9, 0.1],   # "小红"的 K
    [0.0, 0.3, 0.1, 0.5],   # "很"的 K
    [0.7, 0.1, 0.7, 0.0],   # "她"的 K
])

# 第一步：Q 和 K 做点积，算匹配度
scores = Q @ K.T   # 矩阵乘法
print("原始匹配分数：")
print(scores)

# 第二步：softmax 归一化（变成概率，加起来=1）
attention_weights = F.softmax(scores, dim=-1)
print("\n注意力权重（每行加起来=1）：")
print(attention_weights)
print("\n重点看第3行（'她'对每个词的注意力）：")
print(f"  她→小红: {attention_weights[2][0]:.3f}")
print(f"  她→很:   {attention_weights[2][1]:.3f}")
print(f"  她→她:   {attention_weights[2][2]:.3f}")
```

运行后观察：**"她"最关注哪个词？**

输出结果：

```
原始匹配分数：
tensor([[0.4000, 0.1400, 0.3000],
        [0.0400, 0.1300, 0.0100],
        [1.4600, 0.1100, 1.2000]])

注意力权重（每行加起来=1）：
tensor([[0.3737, 0.2881, 0.3381],
        [0.3263, 0.3570, 0.3167],
        [0.4925, 0.1277, 0.3798]])

重点看第3行（'她'对每个词的注意力）：
  她→小红: 0.493
  她→很:   0.128
  她→她:   0.380
```

### `Q @ K.T`

这一行有两个知识点：

**① `K.T` — 矩阵转置**

行和列互换：

```python
K 的形状: (3, 4)    # 3个token，每个4维
K.T 的形状: (4, 3)  # 转置后：4行3列

# 原来的 K:
# [[0.8, 0.2, 0.9, 0.1],
#  [0.0, 0.3, 0.1, 0.5],
#  [0.7, 0.1, 0.7, 0.0]]

# K.T（行变列，列变行）:
# [[0.8, 0.0, 0.7],
#  [0.2, 0.3, 0.1],
#  [0.9, 0.1, 0.7],
#  [0.1, 0.5, 0.0]]
```

**② `@` — 矩阵乘法运算符**

等价于 `torch.matmul(Q, K.T)`，`@` 是它的简写：

```python
Q     @ K.T   = scores
(3,4) @ (4,3) = (3,3)   # 结果是 3×3 的矩阵
```

结果 `scores[i][j]` 就是第 i 个 token 的 Q 和第 j 个 token 的 K 的**点积**（逐元素相乘再求和）：

```python
# 举例：scores[2][0] 就是 "她"的Q 和 "小红"的K 的点积
scores[2][0] = 0.9*0.8 + 0.1*0.2 + 0.8*0.9 + 0.0*0.1
             = 0.72 + 0.02 + 0.72 + 0.0
             = 1.46
# 值越大 → "她"和"小红"越匹配 → 注意力权重越高
```

### `f"  她→小红: {attention_weights[2][0]:.3f}"`

这是 Python 的 **f-string（格式化字符串）**，拆解如下：

```python
f"  她→小红: {attention_weights[2][0]:.3f}"
│             │                        │
│             │                        └── 格式说明符
│             └── 花括号里放 Python 表达式
└── f 开头，表示这是格式化字符串
```

**① `attention_weights[2][0]`**

取矩阵第 2 行第 0 列的值（从 0 开始数）：
- `[2]` → 第 3 个 token（"她"）
- `[0]` → 对第 1 个 token（"小红"）的注意力权重

**② `:.3f`**

冒号后面是格式控制：
- `.3` → 保留 3 位小数
- `f` → 浮点数（float）格式

```python
# 对比效果：
x = 0.123456789

print(f"{x}")      # 输出: 0.123456789（原样）
print(f"{x:.3f}")  # 输出: 0.123（保留3位小数）
print(f"{x:.1f}")  # 输出: 0.1（保留1位小数）
```

## Q、K、V 从哪来？

答案：和 FFN 一样——**矩阵乘法！**

```
回忆 FFN：
  gate    = x @ W₁    ← 同一个 x，乘不同的权重，得到不同的东西
  content = x @ W₃

Attention 完全一样：
  Q = x @ W_Q    ← 同一个 x，乘不同的权重
  K = x @ W_K
  V = x @ W_V
```

**三个 W 是不同的权重矩阵，训练时自动学会的**

---

## 为什么叫 Self Attention
因为：**Self = Q、K、V 全部来自同一个序列，每个 token 回头看整个序列（包括自己），决定该关注谁。**‘

与之对应的是 `Cross Attention`

| 类型 | Q 来自 | K, V 来自 | 用在哪 |
|---|---|---|---|
| **Cross Attention** | 序列 A | 序列 B（另一个序列） | 翻译、图文匹配 |
| **Self Attention** | 序列 A | 序列 A（**自己**） | GPT、LLaMA、MiniMind |

```
          Self Attention              Cross Attention
          ┌─────────┐                ┌─────────────────┐
          │ 序列 A   │                │ 序列 A → Q       │
          │  ↓↓↓    │                │ 序列 B → K, V    │
          │ Q, K, V  │                └─────────────────┘
          │ 全来自 A  │
          └─────────┘
           自己看自己                   A 去看 B
```

MiniMind / GPT / LLaMA 这类 Decoder-Only 模型，用的全部是 Self Attention。 因为它们的任务是：给定前面的文字，预测下一个字。只需要句子自己内部互相看就够了。

# 3. Self-Attention 流程

我们用具体数字走一遍：

```
假设：
  3 个 Token："小红"  "很"  "她"
  每个 Token 的向量维度 dim = 4（真实是 512，这里简化）

输入 x：              3 个 Token × 4 维
W_Q, W_K, W_V：       4 × 4 的矩阵（可训练）
```

## 1：生成 Q、K、V

```
Q = x @ W_Q     # [3, 4] × [4, 4] = [3, 4]
K = x @ W_K     # [3, 4] × [4, 4] = [3, 4]
V = x @ W_V     # [3, 4] × [4, 4] = [3, 4]

每个 Token 都有了自己的 Q、K、V 向量
```

## 2：Q × K^T 算匹配度

```
scores = Q @ K^T    # [3, 4] × [4, 3] = [3, 3]

结果是 3×3 的矩阵：
            小红   很    她
  小红  [  0.8   0.1   0.3 ]  ← 小红和谁相关？
  很    [  0.2   0.5   0.1 ]  ← 很和谁相关？
  她    [  0.9   0.1   0.7 ]  ← 她和谁相关？
```

## 3：除以 √d（缩放）

```
scores = scores / √4 = scores / 2

为什么要除？→ 防止数字太大
```

这个等下细讲，先走完整个流程。

## 4：Softmax 变成概率

```
attention_weights = softmax(scores)

            小红   很    她
  小红  [  0.50  0.20  0.30 ]  ← 每行加起来 = 1
  很    [  0.25  0.50  0.25 ]
  她    [  0.65  0.10  0.25 ]  ← "她"65%注意力给了"小红" ✅
```

## 5：用权重加权 V

```
output = attention_weights @ V    # [3, 3] × [3, 4] = [3, 4]

对"她"来说：
  output["她"] = 0.65 × V["小红"] + 0.10 × V["很"] + 0.25 × V["她"]

"她"的输出向量主要包含了"小红"的信息！
```

## 全流程一图总结

```
x (输入)
│
├── × W_Q → Q ─────┐
├── × W_K → K ───┐ │
└── × W_V → V ─┐ │ │
                │ │ │
                │ ▼ ▼
                │ Q × K^T → scores
                │    │
                │    ▼
                │  ÷ √d
                │    │
                │    ▼
                │  softmax → 注意力权重
                │    │
                ▼    ▼
              weights × V → 输出（融合了其他 Token 信息的新向量）
```

## 为什么要除以 √d？

```
假设 dim = 512

Q 和 K 做点积时，是 512 个数字分别相乘再加起来
→ 结果的数值会很大（几百甚至上千）

大数字进 softmax 会怎样？
```

```python
import torch.nn.functional as F
import torch

# 数字小的时候
small = torch.tensor([1.0, 2.0, 3.0])
print(F.softmax(small, dim=0))
# → [0.09, 0.24, 0.67]  ← 分布比较均匀

# 数字大的时候
big = torch.tensor([10.0, 20.0, 30.0])
print(F.softmax(big, dim=0))
# → [0.00, 0.00, 1.00]  ← 几乎全给了最大的！
```

```
数字太大 → softmax 变成 "赢者通吃"
→ 只关注一个词，完全忽略其他词
→ 模型变得很极端，学不好

除以 √d 把数字缩小到合理范围 → softmax 分布更平滑
```

## ✅ 动手实操

把完整的 Self-Attention 手动实现一遍：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# 超参数
dim = 4          # 向量维度（简化版，真实是512）
seq_len = 3      # 3个Token

# 模拟输入：3个Token，每个4维
x = torch.tensor([
    [1.0, 0.0, 1.0, 0.0],   # Token 0: "小红"
    [0.0, 1.0, 0.0, 1.0],   # Token 1: "很"
    [1.0, 0.1, 0.9, 0.0],   # Token 2: "她"（和"小红"很像！）
])

# 创建 W_Q, W_K, W_V（可训练的权重）
W_Q = nn.Linear(dim, dim, bias=False)
W_K = nn.Linear(dim, dim, bias=False)
W_V = nn.Linear(dim, dim, bias=False)

# 第1步：生成 Q, K, V
Q = W_Q(x)    # [3, 4]
K = W_K(x)    # [3, 4]
V = W_V(x)    # [3, 4]
print("Q 的形状：", Q.shape)
print("K 的形状：", K.shape)
print("V 的形状：", V.shape)

# 第2步：Q × K^T 算匹配度
scores = Q @ K.T    # [3, 3]
print("\n原始匹配分数：")
print(scores)

# 第3步：除以 √d
scores = scores / (dim ** 0.5)
print("\n缩放后的分数：")
print(scores)

# 第4步：softmax
attention_weights = F.softmax(scores, dim=-1)
print("\n注意力权重：")
print(attention_weights)
print("\n'她'(Token2) 对每个词的注意力：")
print(f"  她→小红: {attention_weights[2][0]:.3f}")
print(f"  她→很:   {attention_weights[2][1]:.3f}")
print(f"  她→她:   {attention_weights[2][2]:.3f}")

# 第5步：加权求和
output = attention_weights @ V    # [3, 3] × [3, 4] = [3, 4]
print("\n输出形状：", output.shape)
print("输入和输出形状相同！都是 [3, 4]")
```

运行后观察两件事：
1. **"她"最关注谁？** （注意力权重最大的那个）
2. **输入和输出的形状** 是否一样？

输出结果：

```
Q 的形状： torch.Size([3, 4])
K 的形状： torch.Size([3, 4])
V 的形状： torch.Size([3, 4])

原始匹配分数：
tensor([[-0.0169,  0.1395, -0.0236],
        [-0.3668, -0.1999, -0.3604],
        [-0.0078,  0.1269, -0.0131]], grad_fn=<MmBackward0>)

缩放后的分数：
tensor([[-0.0085,  0.0698, -0.0118],
        [-0.1834, -0.0999, -0.1802],
        [-0.0039,  0.0635, -0.0066]], grad_fn=<DivBackward0>)

注意力权重：
tensor([[0.3249, 0.3513, 0.3238],
        [0.3236, 0.3518, 0.3246],
        [0.3261, 0.3488, 0.3252]], grad_fn=<SoftmaxBackward0>)

'她'(Token2) 对每个词的注意力：
  她→小红: 0.326
  她→很:   0.349
  她→她:   0.325

输出形状： torch.Size([3, 4])
输入和输出形状相同！都是 [3, 4]
```

---

## 为什么输入相近的向量不是最关注的？

> 上面代码里，"她" 的输入向量 `[1.0, 0.1, 0.9, 0.0]` 故意设计得和
> "小红" `[1.0, 0.0, 1.0, 0.0]` 很接近。
>
> 但运行后，"她"不一定最关注"小红"。为什么？
>
> 提示：Q 和 K 是直接用 x 算匹配度的吗？中间经过了什么？

答案：因为中间经过了 W_Q 和 W_K 的变换！

# 4. Causal Mask（掩码机制）

```
假设模型正在生成："今天 天气 很 好"

当模型处理到"很"的时候：
  ✅ 能看到："今天""天气""很"（已经出现的）
  ❌ 不能看到："好"（还没生成呢！）
```

但我们上一节的 Self-Attention，**每个 Token 都能看到所有 Token**：

```
注意力权重矩阵（无掩码）：

          今天  天气   很   好
今天  [  0.4   0.3  0.2  0.1 ]  ← 看到了"好"？不对！
天气  [  0.2   0.4  0.2  0.2 ]  ← 看到了"好"？不对！
很    [  0.1   0.3  0.4  0.2 ]  ← 看到了"好"？不对！
好    [  0.1   0.2  0.3  0.4 ]
```

这就是**信息泄露**——模型作弊偷看了答案。

## 如何解决信息泄露
方案：加一个三角形掩码
> 备注：三角掩码不是标准术语，那只是在描述它的形状，实际上说的就是 Causal Mask

```
Causal Mask：

          今天  天气   很   好
今天  [   ok   ❌   ❌   ❌ ]  ← 只能看自己
天气  [   ok   ok   ❌   ❌ ]  ← 只能看到"今天"和自己
很    [   ok   ok   ok   ❌ ]  ← 看不到"好"
好    [   ok   ok   ok   ok ]  ← 最后一个，全都能看
```

```
具体做法：把 ❌ 的位置设为 -∞（负无穷）

scores（掩码后）：
          今天  天气   很    好
今天  [  0.8   -∞    -∞   -∞  ]
天气  [  0.5   0.6   -∞   -∞  ]
很    [  0.3   0.7   0.4  -∞  ]
好    [  0.2   0.5   0.3  0.8 ]

经过 softmax：
  e^(-∞) = 0  ← 被掩盖的位置权重直接归零！

          今天  天气   很    好
今天  [  1.00  0.00  0.00  0.00 ]
天气  [  0.47  0.53  0.00  0.00 ]
很    [  0.24  0.47  0.29  0.00 ]  ← "好"的权重 = 0 ✅
好    [  0.13  0.27  0.22  0.38 ]
```

## 为什么叫 Causal？

```
Causal = 因果性，核心含义是"有方向的时间关系"

  过去 → 现在 → 未来

  "今天" 可以影响 "天气" 的生成（过去影响现在）
  "天气" 不能影响 "今天" 的生成（未来不能影响过去）

这跟物理学里的因果律一样：原因必须在结果之前

所以 Causal Mask = 只允许看到过去，禁止偷看未来
```

## 哪些模型用 Causal Mask？

```
Decoder-Only 架构（生成式模型）全都用：
  GPT 系列 ✅
  LLaMA    ✅
  DeepSeek ✅
  MiniMind ✅

Encoder 架构（理解式模型）不用：
  BERT ❌ → 它用的是双向注意力，每个词都能看到全文

区分：
  要生成文字 → 必须用 Causal Mask（不能偷看未来）
  只做理解   → 不需要（完形填空需要看到上下文）
```

---

## 代码实现

### 第 1 步：理解掩码的形状

```python
import torch

seq_len = 4  # 4个Token

# torch.triu = 取上三角，diagonal=1 表示不包含对角线
mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
print("Mask（True = 被掩盖，看不到）：")
print(mask)
# [[False,  True,  True,  True],   ← Token0 只能看 Token0
#  [False, False,  True,  True],   ← Token1 能看 Token0,1
#  [False, False, False,  True],   ← Token2 能看 Token0,1,2
#  [False, False, False, False]]   ← Token3 全都能看
```

### 第 2 步：看 -∞ 经过 softmax 的效果

```python
import torch.nn.functional as F

# 不加掩码
scores_normal = torch.tensor([1.0, 2.0, 3.0])
print("正常 softmax：", F.softmax(scores_normal, dim=0))
# → [0.09, 0.24, 0.67]  每个都有权重

# 加掩码：把第3个位置设为 -∞
scores_masked = torch.tensor([1.0, 2.0, float('-inf')])
print("掩码 softmax：", F.softmax(scores_masked, dim=0))
# → [0.27, 0.73, 0.00]  第3个被完全屏蔽！
```

### 第 3 步：完整 Self-Attention + Causal Mask

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def causal_self_attention(x, W_Q, W_K, W_V):
    """带 Causal Mask 的 Self-Attention"""
    seq_len, dim = x.shape
  
    # 生成 Q, K, V
    Q = W_Q(x)
    K = W_K(x)
    V = W_V(x)
  
    # Q × K^T → 匹配度
    scores = Q @ K.T                          # [seq_len, seq_len]
  
    # 缩放
    scores = scores / (dim ** 0.5)
  
    # ⭐ Causal Mask：遮盖未来位置
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    scores = scores.masked_fill(mask, float('-inf'))
  
    # softmax
    weights = F.softmax(scores, dim=-1)
  
    # 加权求和
    output = weights @ V
  
    return output, weights

# 测试
dim = 4
x = torch.randn(5, dim)   # 5个Token
W_Q = nn.Linear(dim, dim, bias=False)
W_K = nn.Linear(dim, dim, bias=False)
W_V = nn.Linear(dim, dim, bias=False)

output, weights = causal_self_attention(x, W_Q, W_K, W_V)

print("注意力权重：")
print(weights.detach().round(decimals=2))
print()

# 验证：右上角是否全为 0
print("验证 Causal Mask 生效：")
for i in range(5):
    for j in range(i+1, 5):
        assert weights[i][j].item() == 0.0
print("✅ 所有未来位置的权重都是 0！")

# 验证：每行加起来是否为 1
print("\n每行权重之和：")
for i in range(5):
    print(f"  Token{i}: {weights[i].sum().item():.4f}")
```

运行上面 3 段代码，确认：
1. 掩码矩阵是**上三角** True
2. `-inf` 经过 softmax 变成 **0**
3. 注意力权重**右上角全是 0**，每行**加起来为 1**

> 我们现在的 Self-Attention 只有**一组** W_Q、W_K、W_V。
> 就好比只有**一个视角**去理解句子。
>
> "她很开心"——"她"和前文的关系有多种：
> - 语法层面："她"指代"小红"
> - 情感层面："开心"的情绪关联
> - 位置层面：相邻词的搭配习惯
>
> 一组 Q、K、V 只能学到**一种**关系模式。
> 如果用**多组**不同的 W_Q、W_K、W_V，同时捕捉不同关系呢？
