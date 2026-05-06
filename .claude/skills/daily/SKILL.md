---
name: daily
description: 生成当日的 daily 和/或 notes 模板文件。无参数创建 daily，传 note 创建笔记，传 all 同时创建两者。
disable-model-invocation: true
allowed-tools: Bash(mkdir *), Bash(date *), Write, Read, Glob
---

# 生成每日模板文件

## 当前日期信息
!`date "+YEAR=%Y MONTH=%m DAY=%d WEEK=%U DOW=%a"`

## 参数

- 无参数或 `daily`：只创建 daily 文件
- `note`：只创建 notes 文件
- `all`：同时创建 daily 和 notes 文件

## 规则

根据上面的日期信息，按以下步骤操作（注意：DOW 值需转为小写，如 Fri → fri）：

1. 计算目录路径：`src/content/daily/{YEAR}/{MONTH}/w{WEEK}/` 和 `src/content/notes/{YEAR}/{MONTH}/w{WEEK}/`（根据参数决定需要哪些）
2. 文件名格式：`{DAY}-{dow}.md`（如 `20-fri.md`，dow 为小写）
3. 用 `mkdir -p` 创建所需目录
4. **如果文件已存在则跳过，告知用户**
5. 根据参数创建对应文件，内容如下：

### daily 文件模板
```markdown
---
title: 日记
date: "{YEAR}-{MONTH}-{DAY}"
tags: [daily]
---

# 学习阶段

# 学习内容

```

### notes 文件模板
```markdown
---
title: 待填写
date: "{YEAR}-{MONTH}-{DAY}"
tags: [note]
---

```

6. 完成后告知用户已创建的文件路径
