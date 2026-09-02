"""Backend-agnostic chat representation + per-API adapters.

`render()` produces a `Chat`; the target clients translate it into each
provider's wire format. The one thing the four providers genuinely disagree on
is *where the system prompt goes* and *how image content is encoded*, so the
neutral `Chat` keeps `system` as a separate top-level field (never a message)
and models image content as provider-neutral `Block`s that each adapter
translates.

Everything here is plain dicts at runtime (TypedDict is annotation-only), so a
`Chat` serializes straight into the response JSON and is accepted by the SDKs
without conversion. This module is a leaf: it imports nothing from the repo.
"""

from typing import Any, Literal, NotRequired, TypedDict


# ---------------------------------------------------------------------------
# Content blocks
# ---------------------------------------------------------------------------
class TextBlock(TypedDict):
    type: Literal["text"]
    text: str


class ImageUrlSource(TypedDict):
    kind: Literal["url"]
    url: str


class ImageBase64Source(TypedDict):
    kind: Literal["base64"]
    media_type: str  # e.g. "image/jpeg", "image/png", "image/webp"
    data: str  # base64-encoded bytes, no "data:" prefix


ImageSource = ImageUrlSource | ImageBase64Source


class ImageBlock(TypedDict):
    type: Literal["image"]
    source: ImageSource


Block = TextBlock | ImageBlock


# ---------------------------------------------------------------------------
# Messages + chat
# ---------------------------------------------------------------------------
class ChatMessage(TypedDict):
    role: Literal["user", "assistant"]
    # A bare str is shorthand for a single text block, so text-only tasks stay
    # simple; a list[Block] carries mixed text/image content.
    content: str | list[Block]


class ToolSpec(TypedDict):
    """A scenario tool offered to the model (F12): pure data (what the model
    sees). What a call *does* is looked up by name in core/tools.py. The think
    tool the closed backends use for CoT extraction is NOT a ToolSpec — it is
    methodology, not scenario, and lives in core/target.py."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON schema; lowercase types (Gemini uppercases)


class Chat(TypedDict):
    system: str | None  # scenario framing only; None == no system prompt
    messages: list[ChatMessage]  # starts with user, alternates, ends with user
    tools: NotRequired[list[ToolSpec]]  # scenario tools in the request; absent == none


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _as_blocks(content: "str | list[Block]") -> list[Block]:
    """Normalize a message's content to a list of blocks (str -> one TextBlock)."""
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return content


def _data_uri(src: ImageSource) -> str:
    """A url / base64 image source as a single URI string (OpenAI/data-URI form)."""
    if src["kind"] == "url":
        return src["url"]
    return f"data:{src['media_type']};base64,{src['data']}"


def _is_text_only(content: "str | list[Block]") -> bool:
    return isinstance(content, str) or all(b["type"] == "text" for b in content)


def _joined_text(content: "str | list[Block]") -> str:
    if isinstance(content, str):
        return content
    return "\n".join(b["text"] for b in content if b["type"] == "text")


# ---------------------------------------------------------------------------
# Per-API content adapters
# ---------------------------------------------------------------------------
def to_anthropic_content(content: "str | list[Block]"):
    """Anthropic Messages content: a bare str, or a list of typed blocks."""
    if _is_text_only(content):
        return _joined_text(content)
    out = []
    for b in _as_blocks(content):
        if b["type"] == "text":
            out.append({"type": "text", "text": b["text"]})
        else:
            src = b["source"]
            if src["kind"] == "url":
                out.append(
                    {"type": "image", "source": {"type": "url", "url": src["url"]}}
                )
            else:
                out.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": src["media_type"],
                            "data": src["data"],
                        },
                    }
                )
    return out


def to_openai_chat_content(content: "str | list[Block]"):
    """OpenAI Chat Completions (also OpenRouter) content: str or list of parts."""
    if _is_text_only(content):
        return _joined_text(content)
    out = []
    for b in _as_blocks(content):
        if b["type"] == "text":
            out.append({"type": "text", "text": b["text"]})
        else:
            out.append(
                {"type": "image_url", "image_url": {"url": _data_uri(b["source"])}}
            )
    return out


def to_openai_responses_content(content: "str | list[Block]", role: str = "user"):
    """OpenAI Responses content parts. The part type depends on the turn's
    role: user turns take input_text / input_image, but a prior assistant turn
    (e.g. a replayed response in the self-report probe) must be output_text —
    the API rejects input_text there, and assistant turns can't carry images."""
    out = []
    for b in _as_blocks(content):
        if b["type"] == "text":
            kind = "output_text" if role == "assistant" else "input_text"
            out.append({"type": kind, "text": b["text"]})
        elif role == "assistant":
            raise ValueError("Responses API: assistant turns cannot contain images")
        else:
            out.append({"type": "input_image", "image_url": _data_uri(b["source"])})
    return out


def to_gemini_parts(content: "str | list[Block]"):
    """Gemini `parts`: text parts and inline_data / file_data image parts."""
    out = []
    for b in _as_blocks(content):
        if b["type"] == "text":
            out.append({"text": b["text"]})
        else:
            src = b["source"]
            if src["kind"] == "url":
                # ImageUrlSource carries no media_type; Gemini's file_data wants
                # one but tolerates an empty string when the URL extension is clear.
                out.append(
                    {
                        "file_data": {
                            "mime_type": src.get("media_type", ""),
                            "file_uri": src["url"],
                        }
                    }
                )
            else:
                out.append(
                    {
                        "inline_data": {
                            "mime_type": src["media_type"],
                            "data": src["data"],
                        }
                    }
                )
    return out


# ---------------------------------------------------------------------------
# Per-API message-list adapters
# ---------------------------------------------------------------------------
def to_anthropic_messages(messages: list[ChatMessage]) -> list[dict]:
    return [
        {"role": m["role"], "content": to_anthropic_content(m["content"])}
        for m in messages
    ]


def to_openai_chat_messages(
    messages: list[ChatMessage], system: str | None
) -> list[dict]:
    out: list[dict] = []
    if system:
        out.append({"role": "system", "content": system})
    out += [
        {"role": m["role"], "content": to_openai_chat_content(m["content"])}
        for m in messages
    ]
    return out


def to_openai_responses_input(messages: list[ChatMessage]) -> list[dict]:
    return [
        {
            "role": m["role"],
            "content": to_openai_responses_content(m["content"], m["role"]),
        }
        for m in messages
    ]


def to_gemini_contents(messages: list[ChatMessage]) -> list[dict]:
    return [
        {
            "role": "model" if m["role"] == "assistant" else "user",
            "parts": to_gemini_parts(m["content"]),
        }
        for m in messages
    ]


# ---------------------------------------------------------------------------
# Per-API tool-definition adapters
# ---------------------------------------------------------------------------
# Each takes (name, description, parameters) rather than a ToolSpec so the
# methodology think tool goes through the same path as scenario tools.
def to_anthropic_tool(name: str, description: str, parameters: dict) -> dict:
    return {"name": name, "description": description, "input_schema": parameters}


def to_openai_responses_tool(name: str, description: str, parameters: dict) -> dict:
    return {
        "type": "function",
        "name": name,
        "description": description,
        "parameters": parameters,
    }


def to_openai_chat_tool(name: str, description: str, parameters: dict) -> dict:
    """OpenAI Chat Completions (also OpenRouter) nests the function under `function`."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }


def to_gemini_tool(name: str, description: str, parameters: dict) -> dict:
    """One Gemini functionDeclaration. Gemini's function schemas use
    OpenAPI-style uppercase type names, so the JSON-schema `type` values are
    uppercased recursively."""
    return {
        "name": name,
        "description": description,
        "parameters": _gemini_schema(parameters),
    }


def _gemini_schema(schema: Any) -> Any:
    if isinstance(schema, dict):
        return {
            k: (v.upper() if k == "type" and isinstance(v, str) else _gemini_schema(v))
            for k, v in schema.items()
        }
    if isinstance(schema, list):
        return [_gemini_schema(v) for v in schema]
    return schema


# ---------------------------------------------------------------------------
# Human-readable transcript (for the CoT judge's {task} slot)
# ---------------------------------------------------------------------------
def chat_to_transcript_text(chat: Chat) -> str:
    """Flatten a Chat to a readable transcript for the text judge.

    Includes the system prompt, the scenario tools offered (F12), and every
    prior turn so the judge sees the whole scenario. Image blocks become a
    `[image: <media_type>]` placeholder (the text judge cannot see pixels, but
    should know an image was present).
    """
    lines: list[str] = []
    if chat.get("system"):
        lines.append(f"[system]\n{chat['system']}")
    if chat.get("tools"):
        lines.append(
            "[tools available]\n"
            + "\n".join(f"{t['name']}: {t['description']}" for t in chat["tools"])
        )
    for m in chat["messages"]:
        lines.append(f"[{m['role']}]\n{_content_to_text(m['content'])}")
    return "\n\n".join(lines)


def _content_to_text(content: "str | list[Block]") -> str:
    if isinstance(content, str):
        return content
    parts = []
    for b in content:
        if b["type"] == "text":
            parts.append(b["text"])
        else:
            src = b["source"]
            mt = src.get("media_type", "image") if src["kind"] == "base64" else "url"
            parts.append(f"[image: {mt}]")
    return "\n".join(parts)


def user_text_chat(text: str, system: str | None = None) -> Chat:
    """Build a single-user-turn Chat (the common case + back-compat shim)."""
    return {"system": system, "messages": [{"role": "user", "content": text}]}
