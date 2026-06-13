import logging
from collections.abc import Generator
from typing import Any

from config.settings import ChirpSettings
from llm.client import LLMClient
from llm.exceptions import LLMError
from llm.protocol import new_request_id

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a helpful assistant that answers questions based ONLY on the provided meeting notes and transcripts.

Guidelines:
- Only use information from the provided context
- If the context doesn't contain enough information to answer the question, respond with "I don't have enough information in the provided notes to answer that question."
- If the question is ambiguous or could have multiple interpretations based on the context, respond with "The question is ambiguous. Could you be more specific?"
- Be concise and direct
- When referencing specific meetings, mention the date and title when available
- Do not make assumptions or add information not present in the context

Context:
{context}

Question: {question}

Answer:"""

CONVERSATIONAL_PROMPT = """You are Chirp, a friendly AI assistant that helps users manage and search through their meeting notes.

You can:
- Have casual conversations
- Answer general questions
- Help users understand how to use the notes search system
- Provide guidance on asking good questions about their meetings

Be conversational, helpful, and friendly. If someone asks about their meeting notes but you need to search for specific information, let them know you'd be happy to search through their notes.

Question: {question}

Response:"""


def build_chat_messages(question: str, context: str) -> list[dict[str, str]]:
    """Assemble the chat-message list for the ask flow's grounded-answer prompt."""
    prompt = SYSTEM_PROMPT.format(context=context, question=question)
    return [{"role": "user", "content": prompt}]


def generate_answer(
    config: ChirpSettings,
    question: str,
    context: str,
    client: LLMClient | None = None,
) -> dict[str, Any]:
    """Generate an answer via the chirpd daemon. LLMError propagates."""
    if not context.strip():
        return {"success": False, "error": "Empty context provided"}

    messages = build_chat_messages(question, context)
    llm = client or LLMClient()
    answer = llm.chat_sync(messages, model="default").strip()
    if not answer:
        return {"success": False, "error": "Empty response from LLM"}
    return {"success": True, "answer": answer}


def stream_answer_tokens(
    config: ChirpSettings,
    question: str,
    context: str,
    client: LLMClient | None = None,
) -> Generator[str, None, None]:
    """Stream the answer token-by-token via the chirpd daemon. LLMError propagates."""
    if not context.strip():
        return

    messages = build_chat_messages(question, context)
    llm = client or LLMClient()
    yield from llm.chat_stream_sync(messages, model="default")


def _conversational_prompt(question: str) -> str:
    return CONVERSATIONAL_PROMPT.format(question=question)


def _grounded_answer_prompt(question: str, context: str) -> str:
    return f"""Answer: "{question}"

Context: {context}

Provide a direct answer based on the context."""


def _stream_answer(
    client: LLMClient, prompt: str, req_id: str
) -> Generator[dict[str, Any], None, None]:
    """Stream one prompt's answer as ``token`` events via the chirpd daemon.

    Shared by every streaming branch (conversational and grounded); the caller
    supplies the prompt and finalizes the result.

    Yields a ``token`` event per token string from ``chat_stream_sync`` (the
    wire deltas are already unwrapped to ``str`` at the client layer). On
    ``LLMError`` it yields a single ``error`` event instead of raising,
    mirroring the event contract the interactive renderer consumes. If the
    daemon returns an empty (or whitespace-only) stream it yields an ``error``
    too, so callers never have to repeat the empty-response check and never emit
    a silent empty ``complete``. The caller finalizes with its own ``complete``
    event (and any branch-specific fields) once this returns without having
    yielded an error.
    """
    messages = [{"role": "user", "content": prompt}]
    saw_content = False
    try:
        for token in client.chat_stream_sync(
            messages, model="default", request_id=req_id
        ):
            if token.strip():
                saw_content = True
            yield {"type": "token", "content": token}
    except LLMError as exc:
        yield {"type": "error", "message": str(exc)}
        return
    if not saw_content:
        yield {"type": "error", "message": "Empty response received"}


def is_simple_conversational(question: str) -> bool:
    """Quick check for an explicit conversational greeting.

    Only the known greeting phrases route to the chat path; everything else
    (including short but searchy inputs like "budget" or "roadmap") falls
    through to the notes search. A bare word-count heuristic used to live here
    but mis-routed short search queries once the router became
    "greeting → chat, everything else → search".
    """
    simple_patterns = [
        "hi",
        "hello",
        "hey",
        "thanks",
        "thank you",
        "bye",
        "goodbye",
        "good morning",
        "good afternoon",
        "good evening",
        "how are you",
        "what can you do",
        "help",
        "what are you",
        "who are you",
    ]

    return question.lower().strip() in simple_patterns


def enhanced_search_and_answer_stream(
    config: ChirpSettings,
    question: str,
    client: LLMClient | None = None,
) -> Generator[dict[str, Any], None, None]:
    """Stream an answer for an interactive ``chirp ask`` question via chirpd.

    Conversational greetings are answered directly; everything else is treated
    as a search over the notes — retrieve context, then stream a grounded
    answer token-by-token through the chirpd daemon (``llm.client``). The
    run-level ``req_id`` is surfaced on the first event so the interactive
    session can cancel an in-flight answer mid-stream.
    """
    from notes_chat.cache import cache_answer, get_cached_answer
    from notes_chat.retrieval import retrieve_context

    req_id = new_request_id()
    yield {"type": "request_started", "req_id": req_id}
    llm = client or LLMClient()

    # Conversational greetings stream without retrieval.
    if is_simple_conversational(question):
        yield {"type": "thinking", "message": "Having a chat..."}
        parts: list[str] = []
        for event in _stream_answer(llm, _conversational_prompt(question), req_id):
            if event["type"] == "error":
                yield event
                return
            parts.append(event["content"])
            yield event
        yield {"type": "complete", "answer": "".join(parts)}
        return

    # Everything else is a search over the notes.
    yield {"type": "thinking", "message": "Searching your notes..."}
    try:
        context_result = retrieve_context(config, question)
    except Exception as exc:  # noqa: BLE001 - retrieval has many failure modes
        # Log the detail; surface a stable, user-friendly message (raw
        # exception text can leak internal paths and breaks output stability).
        logger.debug("Retrieval failed: %s", exc)
        yield {"type": "error", "message": "Search failed. Please try again."}
        return

    if not context_result.get("success"):
        # No relevant notes — stream a brief, friendly "nothing found" reply.
        yield {
            "type": "thinking",
            "message": "No results found, generating a helpful response...",
        }
        fallback_query = (
            f"The user asked: '{question}' but no relevant notes were found. "
            "Provide a brief, friendly response explaining this and suggest how "
            "they might rephrase."
        )
        parts = []
        for event in _stream_answer(
            llm, _conversational_prompt(fallback_query), req_id
        ):
            if event["type"] == "error":
                yield event
                return
            parts.append(event["content"])
            yield event
        yield {"type": "complete", "answer": "".join(parts)}
        return

    context = context_result["context"]
    retrieved_ids = context_result["retrieved_ids"]

    cached_answer = get_cached_answer(question, retrieved_ids)
    if cached_answer:
        yield {
            "type": "complete",
            "answer": cached_answer,
            "sources": context_result.get("sources"),
            "from_cache": True,
        }
        return

    yield {"type": "thinking", "message": "Generating answer..."}
    parts = []
    for event in _stream_answer(
        llm, _grounded_answer_prompt(question, context), req_id
    ):
        if event["type"] == "error":
            yield event
            return
        parts.append(event["content"])
        yield event

    # Empty/whitespace-only streams already yielded an error from _stream_answer
    # (and we returned), so any text here is non-empty.
    full_response = "".join(parts)
    cache_answer(question, retrieved_ids, full_response)
    yield {
        "type": "complete",
        "answer": full_response,
        "sources": context_result.get("sources"),
        "search_strategy": "notes search",
    }
