# 👻 phantom-bridge

> 多 Agent AI 系统的透明记忆桥接层。  
> 无需插件，无需改代码。Agent 共享记忆——它们自己毫不知情。

## 这是什么？

大多数 AI Agent 都是失忆的。每次对话从零开始。如果你同时跑多个 Agent，它们互相不知道对方学到了什么。

**phantom-bridge** 用一层透明代理解决这个问题：

- 拦截 Agent 发出的每一次 LLM API 调用
- 把相关记忆注入到上下文里
- 转发给真实 API —— Agent 完全不知道发生了什么

不改框架，不装插件。只需要把 Agent 的 API 地址指向代理。

## 核心特性

- **完全透明** — Agent 零改动，只改 baseUrl
- **多 Agent 支持** — Jaccard 相关度筛选，每个 Agent 获取最相关的记忆
- **自动采集** — 监控 Hermes MEMORY.md 变化，自动推送到共享记忆池
- **跨机器** — 走 Tailscale，纯 HTTP，零 SSH 依赖
- **持久运行** — launchd 保活，重启自动恢复
- **Token 高效** — 注入 top-3 相关记忆，硬上限 500 tokens

## 快速开始

### 1. 克隆并安装依赖

\`\`\`bash
git clone https://github.com/dpsblack168-rgb/phantom-bridge
cd phantom-bridge
pip install fastapi uvicorn httpx
\`\`\`

### 2. 启动注入代理

\`\`\`bash
python src/v2-inject-proxy.py
# 监听 :18998
\`\`\`

### 3. 启动记忆桥

\`\`\`bash
python src/msg-bridge.py
# 监听 :8083
\`\`\`

### 4. 把 Agent 指向代理

把 Agent 的 LLM baseUrl 从：
\`\`\`
https://api.deepseek.com/v1
\`\`\`
改成：
\`\`\`
http://你的机器IP:18998/v1
\`\`\`

完成。Agent 现在共享记忆了。

## 已知限制

- **仅支持 OpenAI 兼容格式的 API**（DeepSeek、OpenAI 等）。Anthropic 原生 API 暂不支持，计划在 v0.2 中支持。
- **Jaccard 关键词匹配** — 记忆量小时够用，记忆量大时建议升级到向量检索。

## Roadmap

- [ ] \`v0.2\` — Anthropic API 支持
- [ ] \`v0.2\` — BM25 混合检索
- [ ] \`v0.3\` — 向量语义检索
- [ ] \`v0.4\` — 记忆可视化 Web UI

## License

MIT © 2026
