---
name: daily
description: 生成当日的 daily 和 notes 模板文件
disable-model-invocation: true
allowed-tools: Bash(mkdir *), Bash(date *), Write, Read, Glob
---

# 生成每日模板文件

## 当前日期信息
!`date "+YEAR=%Y MONTH=%m DAY=%d WEEK=%U DOW=%a"`

## 规则

根据上面的日期信息，按以下步骤操作（注意：DOW 值需转为小写，如 Fri → fri）：

1. 计算目录路径：`src/content/daily/{YEAR}/{MONTH}/w{WEEK}/` 和 `src/content/notes/{YEAR}/{MONTH}/w{WEEK}/`
2. 文件名格式：`{DAY}-{dow}.md`（如 `20-fri.md`，dow 为小写）
3. 用 `mkdir -p` 创建目录
4. **如果文件已存在则跳过，告知用户**
5. 创建两个文件，内容如下：

### daily 文件模板
```markdown
---
title: 待填写
date: "{YEAR}-{MONTH}-{DAY}"
tags: [daily]
---

## 专注学习时间

<!-- 格式：HH:MM-HH:MM XhYm -->

## 今日目标

## 学习内容

## 收获与反思
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
