# 👻 phantom-bridge

> 多 Agent AI 系统的透明记忆桥接层。
> 无需插件，无需改代码。多个 Agent 共享记忆——它们自己毫不知情。

## 这是什么？

大多数 AI Agent 都是失忆的。每次对话从零开始。如果你同时跑多个 Agent，它们互相不知道对方学到了什么。

**phantom-bridge** 用一层透明代理解决这个问题：

- 拦截 Agent 发出的每一次 LLM API 调用
- 从族群记忆池中检索相关记忆
- 注入到上下文中
- 转发给真实 API —— Agent 完全不知道发生了什么

不改框架，不装插件。只需要把 Agent 的 API 地址指向代理。

## 架构

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Agent (Roxi)   │────→│                  │────→│   DeepSeek API  │
├─────────────────┤     │                  │     ├─────────────────┤
│  Agent (Roxx)   │────→│  phantom-bridge  │────→│   OpenAI API    │
├─────────────────┤     │   :18998         │     ├─────────────────┤
│  Agent (Rovv)   │────→│                  │────→│  Anthropic API  │
└─────────────────┘     │  ↓ 按模型名路由   │     └─────────────────┘
   (Tailscale)          │  ↓ 注入 top-3     │
                        │    记忆           │
                        └────────┬─────────┘
                                 │
                        ┌────────▼─────────┐
                        │   记忆池          │
                        │ (to-rovv.jsonl)   │
                        │                   │
                        │  bge-m3 向量缓存  │
                        └───────────────────┘
```

## Provider 自动路由

phantom-bridge 根据模型名自动识别目标 API：

| 模型前缀       | 路由到            |
|----------------|-------------------|
| `deepseek-*`   | DeepSeek API     |
| `gpt-*`        | OpenAI API       |
| `claude-*`     | Anthropic API    |
| `anthropic-*`  | Anthropic API    |

无需额外配置——设好模型名，代理自动转发。

## 核心特性

- **完全透明** — Agent 零改动，只改 baseUrl
- **多 Provider** — DeepSeek、OpenAI、Anthropic 全支持，按模型名自动路由
- **语义检索** — 通过 Ollama 调用 bge-m3 做向量嵌入，实现语义级记忆匹配（"股价"查"报价"余弦相似度 0.64，Jaccard 仅 0.02）
- **自动降级** — Ollama 不可用时自动回退到 Jaccard 关键词匹配，服务不中断
- **自动采集** — 监控 Hermes MEMORY.md 变化，自动推送到共享记忆池
- **跨机器** — 走 Tailscale，纯 HTTP，零 SSH 依赖
- **持久运行** — launchd 保活，重启自动恢复
- **Token 高效** — 注入 top-3 相关记忆，硬上限 1000 字符（约 500 tokens）

## 快速开始

```bash
git clone https://github.com/dpsblack168-rgb/phantom-bridge
cd phantom-bridge
pip install fastapi uvicorn httpx numpy
python src/v2-inject-proxy.py
```

把 Agent 的 LLM baseUrl 改成 `http://你的机器IP:18998/v1`。

模型设成 `deepseek-v4-pro`、`gpt-4o` 或 `claude-sonnet-4-6-20250514`——代理自动路由。

## 语义检索

v0.3 将原有的 Jaccard 关键词匹配升级为向量语义检索：

1. 启动时，所有记忆通过 bge-m3（BAAI 多语言模型，中英兼容）做向量嵌入
2. 每次请求实时嵌入查询文本
3. 对缓存记忆逐条计算余弦相似度
4. 返回 top-3 结果注入上下文

**实测效果：** 查询"最近京东方股价怎么样"——匹配到记忆中的"京东方 B 报价"记录，余弦相似度 0.64。同样的查询用旧版 Jaccard 仅得 0.02，因为"股价"和"报价"没有一个共同字符。

当 Ollama 未运行时，代理自动降级为 Jaccard 匹配，零停机。

## 已知限制

- Jaccard 降级模式召回率低于向量模式
- 暂无重复/矛盾记忆去重
- 暂无冲突解决机制

## Roadmap

- [x] v0.1 — 基础代理 + Jaccard 记忆注入
- [x] v0.2 — Anthropic API 支持（格式转换、流式转发）
- [x] v0.3 — 向量语义检索（bge-m3 + 余弦相似度）
- [ ] v0.4 — 记忆去重与冲突解决
- [ ] v0.5 — 记忆可视化 Web UI

## License

MIT © 2026
