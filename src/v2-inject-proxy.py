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
from hashlib import md5

import httpx
import numpy as np
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Config ──────────────────────────────────────────────────────────────────
PORT = 18998
MEMORY_FILE = os.path.expanduser("~/messages/to-rovv.jsonl")
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_VERSION = "2023-06-01"

MAX_INJECT = 3              # top-k memories to inject
MAX_INJECT_CHARS = 1000     # ~500 tokens for Chinese-heavy text

# ── Vector embedding ─────────────────────────────────────────────────────────
OLLAMA_URL = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "bge-m3"
EMBED_TIMEOUT = 5.0         # seconds, keep it tight for fallback speed

# Global embedding cache: {(record_idx, record_text_hash): embedding_vector}
_embed_cache: dict[tuple[int, str], list[float]] = {}
_cache_file_hash = ""       # md5 of the jsonl file content, for cache invalidation

# Read API keys from host-process .env as fallback
HOST_DOTENV = os.path.expanduser("~/host-process/.env")
if os.path.exists(HOST_DOTENV):
    _env_map = {}
    for line in open(HOST_DOTENV):
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            _env_map[k] = v.strip("\"'")
    DEEPSEEK_API_KEY = DEEPSEEK_API_KEY or _env_map.get("DEEPSEEK_API_KEY", "")
    ANTHROPIC_API_KEY = ANTHROPIC_API_KEY or _env_map.get("ANTHROPIC_API_KEY", "")

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


# ── Provider routing ────────────────────────────────────────────────────────
ANTHROPIC_MODELS = {"claude", "anthropic"}  # model starts with these prefixes

def detect_provider(model: str) -> str:
    """Returns 'deepseek' or 'anthropic' based on model name."""
    prefix = model.split("/")[0].lower() if "/" in model else model.split("-")[0].lower()
    if prefix in ANTHROPIC_MODELS:
        return "anthropic"
    return "deepseek"


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


# ── Vector embedding & relevance scoring ──────────────────────────────

def _embed(text: str) -> Optional[list[float]]:
    """Call bge-m3 via Ollama for embedding. Returns None on failure."""
    try:
        resp = httpx.post(
            OLLAMA_URL,
            json={"model": EMBED_MODEL, "prompt": text},
            timeout=EMBED_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json().get("embedding")
    except Exception as e:
        logger.warning(f"[向量] Ollama embedding failed: {e}")
        return None


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    arr_a = np.array(a, dtype=np.float64)
    arr_b = np.array(b, dtype=np.float64)
    dot = np.dot(arr_a, arr_b)
    norm = np.linalg.norm(arr_a) * np.linalg.norm(arr_b)
    if norm == 0:
        return 0.0
    return float(dot / norm)


def _record_text(rec: dict) -> str:
    """Build the text to embed for a memory record."""
    sender = rec.get("from", "")
    text = rec.get("text", "")
    return f"{sender}: {text}" if sender else text


def _compute_file_hash() -> str:
    """MD5 of the memory file content for cache invalidation."""
    if not os.path.exists(MEMORY_FILE):
        return ""
    with open(MEMORY_FILE, "rb") as f:
        return md5(f.read()).hexdigest()


def _rebuild_cache(records: list[dict]) -> None:
    """Pre-compute embeddings for all records and populate the cache."""
    global _embed_cache, _cache_file_hash

    _embed_cache = {}
    total = len(records)

    if total == 0:
        _cache_file_hash = _compute_file_hash()
        logger.info("[向量] 记忆为空，跳过缓存构建")
        return

    for idx, rec in enumerate(records):
        text = _record_text(rec)
        vec = _embed(text)
        if vec is not None:
            cache_key = (idx, text)
            _embed_cache[cache_key] = vec

    _cache_file_hash = _compute_file_hash()
    logger.info(f"[向量] 缓存构建完成: {len(_embed_cache)}/{total} 条向量化")


def _ensure_cache(records: list[dict]) -> None:
    """Rebuild cache if the file has changed since last build."""
    current_hash = _compute_file_hash()
    if current_hash != _cache_file_hash:
        logger.info("[向量] 记忆文件变更，重建缓存")
        _rebuild_cache(records)


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


def _jaccard_score(query_tokens: set[str], rec_tokens: set[str]) -> float:
    """Jaccard similarity between two token sets."""
    if not query_tokens or not rec_tokens:
        return 0.0
    intersection = query_tokens & rec_tokens
    union = query_tokens | rec_tokens
    return len(intersection) / len(union)


def score_memories(query: str, records: list[dict]) -> list[tuple[dict, float]]:
    """Score memories using vector semantic search, with Jaccard fallback.

    Strategy:
    1. Rebuild embedding cache if file changed
    2. Embed the query via bge-m3
    3. If embedding succeeded: cosine similarity against all cached embeddings
    4. If embedding failed: fall back to Jaccard token similarity
    """
    if not query or not records:
        return [(r, 0.0) for r in records]

    # Ensure cache is up-to-date
    _ensure_cache(records)

    # Try vector embedding first
    query_vec = _embed(query)

    if query_vec is not None and _embed_cache:
        # Vector path: cosine similarity
        scored = []
        for (idx, text), vec in _embed_cache.items():
            # Find the corresponding record
            if idx < len(records):
                rec = records[idx]
                score = _cosine_similarity(query_vec, vec)
                scored.append((rec, score))
            else:
                # Record index out of range — cache stale, skip
                pass

        # Add any records not in cache with zero score
        cached_indices = {idx for idx, _ in _embed_cache}
        for idx, rec in enumerate(records):
            if idx not in cached_indices:
                scored.append((rec, 0.0))

        scored.sort(key=lambda x: x[1], reverse=True)
        logger.info(f"[向量] 余弦评分完成 ({len(scored)}条, top={scored[0][1]:.4f})")
        return scored

    # Fallback path: Jaccard
    logger.info("[向量] Ollama不可用，降级到Jaccard")
    query_tokens = _tokenize(query)
    if not query_tokens:
        return [(r, 0.0) for r in records]

    scored = []
    for rec in records:
        text = _record_text(rec)
        rec_tokens = _tokenize(text)
        score = _jaccard_score(query_tokens, rec_tokens)
        scored.append((rec, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    logger.info(f"[Jaccard] 评分完成 ({len(scored)}条, top={scored[0][1]:.4f})")
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


# ── Format conversion: OpenAI ↔ Anthropic ───────────────────────────────────

def _openai_to_anthropic(messages: list, body: dict) -> dict:
    """Convert OpenAI-format request body to Anthropic format.

    Anthropic messages must alternate user/assistant, start with user.
    System message is a top-level field, not in messages array.
    """
    # Separate system message
    system_text = None
    chat_messages = []
    for msg in messages:
        if msg.get("role") == "system":
            # Accumulate system messages
            content = msg.get("content", "")
            if system_text is None:
                system_text = content
            else:
                system_text += "\n" + content
        else:
            chat_messages.append(msg)

    # Convert to Anthropic roles: 'assistant' stays, 'user' stays, drop others
    anthro_msgs = []
    role_map = {"user": "user", "assistant": "assistant"}
    for msg in chat_messages:
        role = msg.get("role", "")
        mapped = role_map.get(role)
        if mapped is None:
            continue  # skip system (already handled), tool, function
        content = msg.get("content", "")
        # Ensure content is string (not array of parts) for simplicity
        if isinstance(content, list):
            text_parts = [p.get("text", "") for p in content if p.get("type") == "text"]
            content = "\n".join(text_parts) if text_parts else " "
        if not content or not content.strip():
            content = " "
        anthro_msgs.append({"role": mapped, "content": content})

    # Anthropic requires messages to start with user
    if anthro_msgs and anthro_msgs[0]["role"] != "user":
        anthro_msgs.insert(0, {"role": "user", "content": "[System context injected]"})

    result = {
        "model": body.get("model", "claude-sonnet-4-20250514").split("/")[-1],
        "max_tokens": body.get("max_tokens", 4096),
        "messages": anthro_msgs,
        "stream": body.get("stream", False),
    }
    if system_text:
        result["system"] = system_text

    # Optional params
    if body.get("temperature") is not None:
        result["temperature"] = body["temperature"]

    return result


def _anthropic_to_openai(resp_data: dict, orig_model: str) -> dict:
    """Convert Anthropic non-streaming response to OpenAI format."""
    # Extract text from content blocks
    content_blocks = resp_data.get("content", [])
    text = ""
    for block in content_blocks:
        if block.get("type") == "text":
            text += block.get("text", "")

    stop_reason = resp_data.get("stop_reason", "end_turn")
    finish_reason_map = {
        "end_turn": "stop",
        "max_tokens": "length",
        "stop_sequence": "stop",
        "tool_use": "tool_calls",
    }
    finish_reason = finish_reason_map.get(stop_reason, "stop")

    usage = resp_data.get("usage", {})
    openai_usage = {
        "prompt_tokens": usage.get("input_tokens", 0),
        "completion_tokens": usage.get("output_tokens", 0),
        "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
    }

    return {
        "id": resp_data.get("id", "msg_unknown"),
        "object": "chat.completion",
        "created": int(datetime.now().timestamp()),
        "model": orig_model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": text,
                },
                "finish_reason": finish_reason,
            }
        ],
        "usage": openai_usage,
    }


def _anthropic_stream_to_openai(event_type: str, data: dict) -> list[bytes]:
    """Convert Anthropic stream SSE event to OpenAI-format SSE bytes.

    Returns list of bytes chunks (each is a complete SSE event).
    Empty list if event should be skipped in OpenAI format.
    """
    chunks = []

    if event_type == "message_start":
        # Emit initial OpenAI skeleton chunk
        msg = data.get("message", {})
        skeleton = {
            "id": msg.get("id", "msg_unknown"),
            "object": "chat.completion.chunk",
            "created": int(datetime.now().timestamp()),
            "model": data.get("model", "unknown"),
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": ""},
                    "finish_reason": None,
                }
            ],
        }
        chunks.append(b"data: " + json.dumps(skeleton, ensure_ascii=False).encode() + b"\n\n")

    elif event_type == "content_block_delta":
        delta = data.get("delta", {})
        if delta.get("type") == "text_delta":
            text = delta.get("text", "")
            if text:
                chunk = {
                    "id": "",
                    "object": "chat.completion.chunk",
                    "created": 0,
                    "model": "",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": text},
                            "finish_reason": None,
                        }
                    ],
                }
                chunks.append(b"data: " + json.dumps(chunk, ensure_ascii=False).encode() + b"\n\n")

    elif event_type == "message_delta":
        delta = data.get("delta", {})
        stop_reason = delta.get("stop_reason")
        if stop_reason:
            finish_reason_map = {
                "end_turn": "stop",
                "max_tokens": "length",
                "stop_sequence": "stop",
                "tool_use": "tool_calls",
            }
            fr = finish_reason_map.get(stop_reason, "stop")
            chunk = {
                "id": "",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "",
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": fr,
                    }
                ],
            }
            chunks.append(b"data: " + json.dumps(chunk, ensure_ascii=False).encode() + b"\n\n")

    elif event_type == "message_stop":
        chunks.append(b"data: [DONE]\n\n")

    # message_start also has usage info but we skip it for streaming

    return chunks

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

    # ── Step 4: 路由转发 ──
    provider = detect_provider(model)
    logger.info(f"[路由] provider={provider} model={model}")

    if provider == "anthropic":
        if not ANTHROPIC_API_KEY:
            logger.error("[转发] ANTHROPIC_API_KEY 未设置")
            raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not configured")

        # Convert to Anthropic format (memory already injected into messages)
        anthro_body = _openai_to_anthropic(messages, body)

        headers = {
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        }
        logger.info(f"[转发] → anthropic/{anthro_body['model']} stream={stream}")

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                if stream:
                    req = client.build_request("POST", ANTHROPIC_URL,
                                               headers=headers, json=anthro_body)
                    resp = await client.send(req, stream=True)
                    resp.raise_for_status()

                    async def anthropic_stream_generator():
                        buffer = b""
                        async for raw_chunk in resp.aiter_bytes():
                            buffer += raw_chunk
                            # Anthropic SSE may split across chunks; parse complete events
                            while b"\n\n" in buffer:
                                event_block, buffer = buffer.split(b"\n\n", 1)
                                event_lines = event_block.decode("utf-8", errors="replace").strip().split("\n")
                                event_type = ""
                                data_str = ""
                                for line in event_lines:
                                    if line.startswith("event: "):
                                        event_type = line[7:].strip()
                                    elif line.startswith("data: "):
                                        data_str = line[6:].strip()
                                if data_str:
                                    try:
                                        data = json.loads(data_str)
                                        oai_chunks = _anthropic_stream_to_openai(event_type, data)
                                        for c in oai_chunks:
                                            yield c
                                    except json.JSONDecodeError:
                                        pass

                        # Flush remaining buffer
                        if buffer:
                            logger.debug(f"[流式] 剩余buffer未处理: {buffer[:100]}")

                    logger.info(f"[转发] ← streaming (anthropic) response started")
                    return Response(
                        content=anthropic_stream_generator(),
                        status_code=resp.status_code,
                        headers={
                            "content-type": "text/event-stream",
                            "cache-control": "no-cache",
                        },
                        media_type="text/event-stream",
                    )
                else:
                    resp = await client.post(
                        ANTHROPIC_URL,
                        headers=headers,
                        json=anthro_body,
                        timeout=120.0,
                    )
                    resp.raise_for_status()
                    result = resp.json()
                    usage = result.get("usage", {})
                    logger.info(
                        f"[转发] ← 200 OK ({resp.elapsed.total_seconds():.1f}s, "
                        f"prompt={usage.get('input_tokens', '?')}, "
                        f"completion={usage.get('output_tokens', '?')})"
                    )
                    return _anthropic_to_openai(result, model)

        except httpx.HTTPStatusError as e:
            logger.error(f"[转发] Anthropic returned {e.response.status_code}: {e.response.text[:500]}")
            raise HTTPException(status_code=e.response.status_code,
                                detail=f"Anthropic error: {e.response.text[:500]}")
        except httpx.TimeoutException:
            logger.error(f"[转发] Anthropic timeout")
            raise HTTPException(status_code=504, detail="Anthropic timeout")
        except Exception as e:
            logger.error(f"[转发] Anthropic error: {e}")
            raise HTTPException(status_code=502, detail=f"Anthropic forward error: {e}")

    else:
        # ── DeepSeek / OpenAI-compatible ──
        if not DEEPSEEK_API_KEY:
            logger.error("[转发] DEEPSEEK_API_KEY 未设置")
            raise HTTPException(status_code=500, detail="DEEPSEEK_API_KEY not configured")

        headers = {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json",
        }

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
    logger.info(f"║  DeepSeek: {DEEPSEEK_URL}")
    logger.info(f"║  Anthropic: {ANTHROPIC_URL}")
    logger.info(f"║  DS Key: {mask_key(DEEPSEEK_API_KEY)}")
    logger.info(f"║  Anth Key: {mask_key(ANTHROPIC_API_KEY)}")
    logger.info(f"║  Embed: {EMBED_MODEL} @ {OLLAMA_URL}")
    logger.info(f"╚══════════════════════════════════════════════╝")

    # ── 启动时预构建向量缓存 ──
    logger.info("[启动] 预构建记忆缓存...")
    records = load_memories()
    if records:
        logger.info(f"[启动] 加载 {len(records)} 条记忆")
        _rebuild_cache(records)
    else:
        logger.info("[启动] 无记忆需缓存")
        _cache_file_hash = _compute_file_hash()

    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
