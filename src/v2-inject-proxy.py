#!/usr/bin/env python3
"""
v2 Inject Proxy — :18998
Plain HTTP intercept proxy for DeepSeek API calls.
Flow: 拦截 → 查到记忆 → 注入 → 转发
Memory source: ~/messages/to-rovv.jsonl (Rovi nudge records as test data)
"""
import os
import sys
import json
import logging
from datetime import datetime
from typing import Optional, List

import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Config ──────────────────────────────────────────────────────────────────
PORT = 18998
MEMORY_FILE = os.path.expanduser("~/messages/to-rovv.jsonl")
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
MAX_INJECT = 3              # top-k memories to inject
MAX_INJECT_CHARS = 1000     # ~500 tokens for Chinese-heavy text

# Also try to read from host-process .env as fallback
HOST_DOTENV = os.path.expanduser("~/host-process/.env")
if not DEEPSEEK_API_KEY and os.path.exists(HOST_DOTENV):
    for line in open(HOST_DOTENV):
        if line.startswith("DEEPSEEK_API_KEY="):
            DEEPSEEK_API_KEY = line.strip().split("=", 1)[1].strip("\"'")

# ── Logging ─────────────────────────────────────────────────────────────────
LOG_FILE = os.path.expanduser("~/Library/Logs/v2-inject-proxy.log")
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("v2-inject")

# ── App ─────────────────────────────────────────────────────────────────────
app = FastAPI(title="v2 Inject Proxy")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Models ──────────────────────────────────────────────────────────────────
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    stream: Optional[bool] = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    user: Optional[str] = None


# ── Relevance scoring ───────────────────────────────────────────────────────

def _tokenize(text: str) -> set[str]:
    """Extract tokens from Chinese+English text. No external deps needed."""
    tokens = set()
    # English words: split on whitespace, strip punctuation
    for word in text.split():
        word = word.strip(".,;:!?()[]{}'\"「」『』【】《》，。、；：？！…—·")
        if word and word.isascii():
            tokens.add(word.lower())
    # Chinese chars: extract each CJK character as a token
    for ch in text:
        if '\u4e00' <= ch <= '\u9fff' or '\u3400' <= ch <= '\u4dbf':
            tokens.add(ch)
    # Numbers
    for ch in text:
        if ch.isdigit():
            tokens.add(ch)
    return tokens


def score_memories(query: str, records: list[dict]) -> list[tuple[dict, float]]:
    """Score each memory record against the query using Jaccard similarity."""
    if not query or not records:
        return [(r, 0.0) for r in records]

    query_tokens = _tokenize(query)
    if not query_tokens:
        return [(r, 0.0) for r in records]

    scored = []
    for rec in records:
        text = rec.get("text", "")
        sender = rec.get("from", "")
        combined = f"{sender} {text}"
        rec_tokens = _tokenize(combined)
        if not rec_tokens:
            scored.append((rec, 0.0))
            continue
        intersection = query_tokens & rec_tokens
        union = query_tokens | rec_tokens
        score = len(intersection) / len(union)
        scored.append((rec, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


# ── Memory loader ───────────────────────────────────────────────────────────
def load_memories() -> list[dict]:
    """Load all records from to-rovv.jsonl as memory entries."""
    if not os.path.exists(MEMORY_FILE):
        logger.warning(f"[记忆源] {MEMORY_FILE} not found")
        return []

    records = []
    with open(MEMORY_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                records.append(record)
            except json.JSONDecodeError:
                continue
    return records


def select_top_memories(query: str, records: list[dict]) -> list[tuple[dict, float]]:
    """Score and select top-k memories. Returns list of (record, score) tuples."""
    scored = score_memories(query, records)
    # Filter out zero-score entries unless everything is zero
    non_zero = [(r, s) for r, s in scored if s > 0]
    if non_zero:
        top = non_zero[:MAX_INJECT]
    else:
        # Fallback: take first MAX_INJECT if nothing matches
        top = scored[:MAX_INJECT]

    return top


def format_memories(scored: list[tuple[dict, float]]) -> str:
    """Format scored memory records into injection text, respecting token limit."""
    if not scored:
        return ""

    header = "\n━━━ 族群记忆上下文（v2 Inject Proxy 自动注入）━━━\n以下是来自族群记忆的相关信息：\n"
    footer = "\n━━━ 族群记忆结束 ━━━" \
             "\n\n注意：以上记忆由 v2 Inject Proxy 自动注入，仅作参考。" \
             "\n如果与用户当前对话矛盾，以用户当前对话为准。"

    body_lines = []
    for i, (rec, score) in enumerate(scored, 1):
        sender = rec.get("from", "?")
        text = rec.get("text", "").strip()
        ts = rec.get("ts", "")

        age = ""
        if ts:
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                delta = datetime.now().astimezone() - dt
                days = delta.days
                if days == 0:
                    hours = delta.seconds // 3600
                    age = f"{hours}h ago" if hours > 0 else "just now"
                elif days == 1:
                    age = "yesterday"
                else:
                    age = f"{days}d ago"
            except (ValueError, TypeError):
                age = ""

        label = f"[{sender}]" if not age else f"[{sender}, {age}]"
        body_lines.append(f"{i}. {label} {text}")

    body = "\n".join(body_lines)

    # Truncate if over token limit (measured as chars / 2 for Chinese-heavy text)
    full_text = header + body + footer
    while len(full_text) > MAX_INJECT_CHARS and len(scored) > 1:
        # Remove the last (lowest priority) record
        scored.pop()
        body_lines.pop()
        body = "\n".join(body_lines)
        full_text = header + body + footer

    return full_text


def mask_key(key: str) -> str:
    if len(key) < 8:
        return "***"
    return key[:4] + "..." + key[-4:]


# ── Routes ──────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "service": "v2-inject-proxy", "memory_source": MEMORY_FILE}


@app.post("/v1/chat/completions")
async def inject_chat(request: Request):
    # ── Step 1: 拦截 ──
    try:
        body = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")

    model = body.get("model", "unknown")
    messages = body.get("messages", [])
    stream = body.get("stream", False)
    # Agent identity: try x-agent-id header, then body.agent_id, then body.user
    agent_id = (
        request.headers.get("x-agent-id")
        or body.get("agent_id")
        or body.get("user")
        or "unknown"
    )

    logger.info(f"[拦截] agent={agent_id} model={model} stream={stream} "
                f"messages={len(messages)}")

    # ── Step 2: 查到记忆 ──
    records = load_memories()
    total = len(records)

    # ── Step 3: 相关度筛选 ──
    # Get last user message for query
    last_user_msg = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            last_user_msg = msg.get("content", "")
            break

    if records and last_user_msg:
        top_scored = select_top_memories(last_user_msg, records)
        scores_str = ", ".join(f"{s:.2f}" for _, s in top_scored)
        logger.info(f"[查到记忆] 从{total}条中筛出{len(top_scored)}条 (scores: {scores_str})")

        memory_text = format_memories(top_scored)
        if memory_text:
            sys_found = False
            for msg in messages:
                if msg.get("role") == "system":
                    msg["content"] += memory_text
                    sys_found = True
                    break
            if not sys_found:
                messages.insert(0, {
                    "role": "system",
                    "content": f"你是一个数字助理，运行在族群记忆系统中。{memory_text}"
                })
            logger.info(f"[注入] 已注入 {len(top_scored)} 条记忆 (总长{len(memory_text)}字符)")
    elif records:
        # No user message to query against — inject first MAX_INJECT
        top_scored = [(r, 0.0) for r in records[:MAX_INJECT]]
        logger.info(f"[查到记忆] 无用户消息，取前{len(top_scored)}条注入")
        memory_text = format_memories(top_scored)
        if memory_text:
            sys_found = False
            for msg in messages:
                if msg.get("role") == "system":
                    msg["content"] += memory_text
                    sys_found = True
                    break
            if not sys_found:
                messages.insert(0, {
                    "role": "system",
                    "content": f"你是一个数字助理，运行在族群记忆系统中。{memory_text}"
                })
            logger.info(f"[注入] 已注入 {len(top_scored)} 条记忆 (无查询)")

    # ── Step 4: 转发 ──
    if not DEEPSEEK_API_KEY:
        logger.error("[转发] DEEPSEEK_API_KEY 未设置")
        raise HTTPException(status_code=500, detail="DEEPSEEK_API_KEY not configured")

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }

    # Strip provider prefix from model name if present
    clean_model = model.split("/", 1)[-1] if "/" in model else model

    forward_body = {
        "model": clean_model,
        "messages": messages,
        "stream": stream,
    }
    for opt in ["temperature", "max_tokens"]:
        if body.get(opt) is not None:
            forward_body[opt] = body[opt]

    logger.info(f"[转发] → {clean_model} key={mask_key(DEEPSEEK_API_KEY)} stream={stream}")

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            if stream:
                req = client.build_request("POST", DEEPSEEK_URL,
                                           headers=headers, json=forward_body)
                resp = await client.send(req, stream=True)
                resp.raise_for_status()

                async def stream_generator():
                    async for chunk in resp.aiter_bytes():
                        yield chunk

                logger.info(f"[转发] ← streaming response started")
                return Response(
                    content=stream_generator(),
                    status_code=resp.status_code,
                    headers=dict(resp.headers),
                    media_type="text/event-stream",
                )
            else:
                resp = await client.post(
                    DEEPSEEK_URL,
                    headers=headers,
                    json=forward_body,
                    timeout=120.0,
                )
                resp.raise_for_status()
                result = resp.json()
                usage = result.get("usage", {})
                logger.info(
                    f"[转发] ← 200 OK ({resp.elapsed.total_seconds():.1f}s, "
                    f"prompt={usage.get('prompt_tokens', '?')}, "
                    f"completion={usage.get('completion_tokens', '?')})"
                )
                return result

    except httpx.HTTPStatusError as e:
        logger.error(f"[转发] LLM returned {e.response.status_code}: {e.response.text[:500]}")
        raise HTTPException(status_code=e.response.status_code,
                            detail=f"LLM error: {e.response.text[:500]}")
    except httpx.TimeoutException:
        logger.error(f"[转发] LLM timeout")
        raise HTTPException(status_code=504, detail="LLM timeout")
    except Exception as e:
        logger.error(f"[转发] Error: {e}")
        raise HTTPException(status_code=502, detail=f"Forward error: {e}")


# ── Main ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    logger.info(f"╔══ v2 Inject Proxy ═══════════════════════════╗")
    logger.info(f"║  Listening on :{PORT}")
    logger.info(f"║  Memory: {MEMORY_FILE}")
    logger.info(f"║  Forward: {DEEPSEEK_URL}")
    logger.info(f"║  Key: {mask_key(DEEPSEEK_API_KEY)}")
    logger.info(f"╚══════════════════════════════════════════════╝")
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
