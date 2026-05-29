# 👻 phantom-bridge

> A transparent memory bridge for multi-agent AI systems.  
> No plugins. No code changes. Your agents share memory — and never notice.

## What is this?

Most AI agent setups are amnesiac. Every session starts from scratch. And if you run multiple agents, they have no idea what each other learned.

**phantom-bridge** solves this with a transparent proxy layer:

- Intercepts every LLM API call your agents make
- Injects relevant shared memories into the context
- Forwards to the real API — agents never know it happened

No framework changes. No plugins to install. Just point your agents at the proxy.

## Key Features

- **Transparent** — zero agent modification required
- **Multi-agent** — Jaccard relevance filtering per agent
- **Self-populating** — auto-captures from Hermes MEMORY.md
- **Cross-machine** — Tailscale, pure HTTP, zero SSH
- **Persistent** — launchd keep-alive
- **Token-efficient** — top-3 injection, 500 token cap

## Quick Start

\`\`\`bash
git clone https://github.com/YOUR_USERNAME/phantom-bridge
cd phantom-bridge
pip install fastapi uvicorn httpx
python src/v2-inject-proxy.py
\`\`\`

Point your agent's base URL at \`http://YOUR_IP:18998/v1\` — done.

## Limitations

- OpenAI-compatible APIs only. Anthropic native API coming in v0.2.

## Roadmap

- [ ] v0.2 Anthropic API support
- [ ] v0.2 BM25 hybrid retrieval  
- [ ] v0.3 Embedding semantic search

## License

MIT © 2026
