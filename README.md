# 👻 phantom-bridge

> A transparent memory bridge for multi-agent AI systems.
> No plugins. No code changes. Your agents share memory — and never notice.

## What is this?

Most AI agent setups are amnesiac. Every session starts from scratch. And if you run multiple agents, they have no idea what each other learned.

phantom-bridge solves this with a transparent proxy layer:

- Intercepts every LLM API call your agents make
- Retrieves relevant shared memories from the family pool
- Injects them into the context
- Forwards to the real API — agents never know it happened

No framework changes. No plugins to install. Just point your agents at the proxy.

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Agent (Roxi)   │────→│                  │────→│   DeepSeek API  │
├─────────────────┤     │                  │     ├─────────────────┤
│  Agent (Roxx)   │────→│  phantom-bridge  │────→│   OpenAI API    │
├─────────────────┤     │   :18998         │     ├─────────────────┤
│  Agent (Rovv)   │────→│                  │────→│  Anthropic API  │
└─────────────────┘     │  ↓ route by      │     └─────────────────┘
   (Tailscale)          │    model name    │
                        │  ↓ inject top-3  │
                        │    memories      │
                        └────────┬─────────┘
                                 │
                        ┌────────▼─────────┐
                        │   memory pool     │
                        │ (to-rovv.jsonl)   │
                        │                   │
                        │  bge-m3 vector    │
                        │  embeddings cache │
                        └───────────────────┘
```

## Provider Auto-Routing

phantom-bridge detects the target provider automatically from the model name:

| Model prefix   | Routes to        |
|----------------|------------------|
| `deepseek-*`   | DeepSeek API     |
| `gpt-*`        | OpenAI API       |
| `claude-*`     | Anthropic API    |
| `anthropic-*`  | Anthropic API    |

No configuration needed — just set the model name and the proxy does the rest.

## Key Features

- **Transparent** — zero agent modification required, just change the base URL
- **Multi-provider** — DeepSeek, OpenAI, and Anthropic all supported, auto-routed by model name
- **Semantic retrieval** — bge-m3 embeddings via Ollama for meaning-aware memory matching (cosine similarity 0.64 for "股价" → "报价", vs 0.02 with Jaccard)
- **Auto-fallback** — if Ollama is unavailable, falls back gracefully to Jaccard keyword matching
- **Self-populating** — auto-captures from Hermes MEMORY.md via nudge-watch
- **Cross-machine** — works over Tailscale, pure HTTP, zero SSH dependencies
- **Persistent** — launchd keeps the proxy alive across reboots
- **Token-efficient** — injects top-3 relevant memories, hard cap at 1000 chars (~500 tokens)

## Quick Start

```bash
git clone https://github.com/dpsblack168-rgb/phantom-bridge
cd phantom-bridge
pip install fastapi uvicorn httpx numpy
python src/v2-inject-proxy.py
```

Point your agent's base URL at `http://YOUR_IP:18998/v1` — done.

Set `model` to `deepseek-v4-pro`, `gpt-4o`, or `claude-sonnet-4-6-20250514` — the proxy routes automatically.

## Semantic Retrieval

v0.3 replaces the original Jaccard keyword matching with vector embeddings:

1. On startup, all memories are embedded via bge-m3 (BAAI multilingual model, supports Chinese + English)
2. Each incoming query is embedded in real time
3. Cosine similarity scores each cached memory against the query
4. Top-3 results are injected

**Real-world example:** A query about "股价" (stock price) scored 0.64 against memories containing "报价" (price quotes) — the old Jaccard method scored only 0.02 because the two words share no common characters.

When Ollama is not running, the proxy logs a warning and automatically degrades to Jaccard matching — zero downtime.

## Limitations

- Jaccard keyword fallback has lower recall than the vector path
- No deduplication of duplicate or contradictory memories yet
- No conflict resolution

## Roadmap

- [x] v0.1 — Initial proxy + memory injection via Jaccard
- [x] v0.2 — Anthropic API support (format conversion, streaming)
- [x] v0.3 — Vector semantic retrieval (bge-m3 + cosine similarity)
- [ ] v0.4 — Memory deduplication & conflict resolution
- [ ] v0.5 — Web UI for memory inspection

## License

MIT © 2026
