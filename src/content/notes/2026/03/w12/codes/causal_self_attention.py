import torch
import torch.nn as nn
import torch.nn.functional as F

def causal_self_attention(x, W_Q, W_K, W_V):
    """带 Causal Mask 的 Self-Attention"""
    seq_len, dim = x.shape

    print("mo 输入序列 x:", x)
    print(f"mo 输入序列长度: {seq_len}, 维度: {dim}")

    # 生成 Q, K, V
    Q = W_Q(x)
    K = W_K(x)
    V = W_V(x)

    print("mo Q, K, V 形状:", Q.shape, K.shape, V.shape)

    # Q × K^T → 匹配度
    scores = Q @ K.T                          # [seq_len, seq_len]

    print("mo 匹配度矩阵形状:", scores.shape)

    # 缩放
    scores = scores / (dim ** 0.5)

    # ⭐ Causal Mask：遮盖未来位置
    # torch.triu 生成上三角矩阵，diagonal=1 表示主对角线以上的元素为 1
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
