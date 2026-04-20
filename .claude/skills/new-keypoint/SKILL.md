---
name: new-keypoint
description: Use when creating a new topic/learning document in keyPoints. Triggers: "新建专题"、"创建学习笔记"、"新增 keyPoint"，或用户要求在 keyPoints 目录下创建新文档时。
---

# 创建 KeyPoint 专题文档

在 `src/content/keyPoints/` 下创建新的 MDX 学习笔记文档。

## 流程

1. 确定主题名称（英文 camelCase + `Learning` 后缀作为文件名）
2. 查看现有文件确定下一个 `order` 值
3. 生成 frontmatter 并创建文件
4. 告知用户文件已创建

## 文件命名

- 格式：`{topicName}Learning.mdx`
- 使用 camelCase：`helloAgentsLearning.mdx`、`minimindLearning.mdx`
- 主题名来自用户描述，取英文关键词

## Frontmatter 模板

```yaml
---
title: {中文标题} 学习笔记
order: {下一个序号}
tags: [{相关标签}]
updatedAt: "{当天日期 YYYY-MM-DD}"
---
```

## 规则

- 默认创建空文档（仅含 frontmatter），除非用户要求添加初始内容
- `order` 值通过查看现有文件中最大的 order + 1 确定
- `tags` 根据主题推断 2-3 个合理标签
- `updatedAt` 使用当天日期
- 不要自动添加 `import` 语句或正文内容，用户会自行填充

## 速查

| 字段 | 规则 |
|------|------|
| 文件名 | camelCase + `Learning.mdx` |
| title | 中文，`{主题} 学习笔记` |
| order | 现有最大值 + 1 |
| tags | 2-3 个相关标签 |
| updatedAt | 当天日期 |
| 正文 | 默认为空 |
