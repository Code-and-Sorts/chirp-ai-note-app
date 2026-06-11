import logging
from collections.abc import Generator
from typing import Any

import requests

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

SEARCH_ORCHESTRATION_PROMPT = """You are an intelligent search orchestrator for meeting notes. Your job is to analyze a user's question and determine the best search strategy.

Given a user question, extract:
1. **Search terms**: Key words/phrases to search for
2. **Time filter**: Any date/time references (like "Sept 17", "yesterday", "last week", etc.)
3. **Search strategy**: What type of information they're looking for

User question: "{question}"

Respond with JSON in this exact format:
{{
    "search_terms": ["term1", "term2", "term3"],
    "time_filter": "extracted_time_reference_or_null",
    "search_strategy": "brief_description_of_what_user_wants",
    "requires_search": true_or_false
}}

Examples:
- "what time was my last meeting on Sept 17?" → {{"search_terms": ["meeting", "time"], "time_filter": "Sept 17", "search_strategy": "find meeting time on specific date", "requires_search": true}}
- "hi there" → {{"search_terms": [], "time_filter": null, "search_strategy": "casual greeting", "requires_search": false}}
- "what did we discuss about the budget?" → {{"search_terms": ["budget", "discuss"], "time_filter": null, "search_strategy": "find budget discussions", "requires_search": true}}
"""


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


def generate_conversational_response(
    config: ChirpSettings, question: str
) -> dict[str, Any]:
    """Generate a conversational response without requiring context."""
    try:
        prompt = CONVERSATIONAL_PROMPT.format(question=question)

        response = requests.post(
            f"{config.models.ollama_url}/api/generate",
            json={
                "model": config.models.llm,
                "prompt": prompt,
                "temperature": 0.3,
                "top_p": 0.9,
                "stream": False,
            },
            timeout=30,
        )

        if response.status_code != 200:
            error_msg = f"Ollama API error: {response.status_code}"
            if response.status_code == 404:
                error_msg += f". Model '{config.models.llm}' not found. Try: ollama pull {config.models.llm}"
            elif response.status_code == 500:
                error_msg += (
                    ". Ollama server error. Is ollama running? Try: ollama serve"
                )
            return {"success": False, "error": error_msg}

        result = response.json()
        answer = result.get("response", "").strip()

        if not answer:
            return {"success": False, "error": "Empty response from LLM"}

        return {"success": True, "answer": answer}

    except requests.exceptions.ConnectionError:
        return {
            "success": False,
            "error": "Cannot connect to Ollama. Is it running? Try: ollama serve",
        }
    except requests.exceptions.Timeout:
        return {
            "success": False,
            "error": "Ollama request timed out.",
        }
    except ConnectionError:
        return {
            "success": False,
            "error": "Cannot connect to Ollama. Is it running? Try: ollama serve",
        }
    except Exception as e:  # noqa: BLE001 - fallback after specific request handlers
        logger.debug("Failed to generate response: %s", e)
        return {"success": False, "error": f"Failed to generate response: {e}"}


def orchestrate_search(config: ChirpSettings, question: str) -> dict[str, Any]:
    """Use LLM to analyze question and determine search strategy."""
    try:
        prompt = SEARCH_ORCHESTRATION_PROMPT.format(question=question)

        response = requests.post(
            f"{config.models.ollama_url}/api/generate",
            json={
                "model": config.models.llm,
                "prompt": prompt,
                "temperature": 0,
                "top_p": 1,
                "stream": False,
            },
            timeout=30,
        )

        if response.status_code != 200:
            return {
                "success": False,
                "error": f"Ollama API error: {response.status_code}",
            }

        result = response.json()
        raw_response = result.get("response", "").strip()

        if not raw_response:
            return {"success": False, "error": "Empty response from LLM"}

        try:
            import json
            import re

            json_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", raw_response)
            if json_match:
                json_str = json_match.group()
                search_plan = json.loads(json_str)
            else:
                search_plan = json.loads(raw_response)

            required_keys = [
                "search_terms",
                "time_filter",
                "search_strategy",
                "requires_search",
            ]
            if not all(key in search_plan for key in required_keys):
                return {"success": False, "error": "Invalid response format from LLM"}

            return {"success": True, "search_plan": search_plan}

        except json.JSONDecodeError:
            return {
                "success": False,
                "error": f"Failed to parse LLM response as JSON: {raw_response}",
            }

    except requests.exceptions.ConnectionError:
        return {
            "success": False,
            "error": "Cannot connect to Ollama. Is it running? Try: ollama serve",
        }
    except Exception as e:  # noqa: BLE001 - fallback after specific request handlers
        logger.debug("Failed to orchestrate search: %s", e)
        return {"success": False, "error": f"Failed to orchestrate search: {e}"}


def _conversational_prompt(question: str) -> str:
    return CONVERSATIONAL_PROMPT.format(question=question)


def _fast_answer_prompt(question: str, context: str) -> str:
    return f"""Answer this question about meeting notes: "{question}"

Context: {context}

Answer:"""


def _grounded_answer_prompt(question: str, context: str) -> str:
    return f"""Answer: "{question}"

Context: {context}

Provide a direct answer based on the context."""


def _stream_answer(
    client: LLMClient, prompt: str, req_id: str
) -> Generator[dict[str, Any], None, None]:
    """Stream one prompt's answer as ``token`` events via the chirpd daemon.

    Shared by every streaming branch (conversational, fast-answer, and grounded);
    the caller supplies the prompt and finalizes the result.

    Yields a ``token`` event per token string from ``chat_stream_sync`` (the
    wire deltas are already unwrapped to ``str`` at the client layer). On
    ``LLMError`` it yields a single
    ``error`` event instead of raising, mirroring the event contract the
    interactive renderer consumes. If the daemon returns an empty (or
    whitespace-only) stream it yields an ``error`` too, so callers never have to
    repeat the empty-response check and never emit a silent empty ``complete``.
    The caller finalizes with its own ``complete`` event (and any branch-specific
    fields) once this returns without having yielded an error.
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
    """Quick check for obvious conversational queries."""
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

    question_lower = question.lower().strip()
    return question_lower in simple_patterns or len(question.split()) <= 2


def is_obvious_search(question: str) -> bool:
    """Quick check for obvious search queries."""
    search_patterns = [
        "what did",
        "who said",
        "when did",
        "what was discussed",
        "tell me about",
        "find",
        "search",
        "show me",
        "what happened",
        "meeting",
        "discussed",
        "action item",
        "decision",
        "summary",
        "agenda",
    ]

    question_lower = question.lower()
    return any(pattern in question_lower for pattern in search_patterns)


def fast_search_and_answer(config: ChirpSettings, question: str) -> dict[str, Any]:
    """Fast path for obvious search queries - skip orchestration."""
    from notes_chat.cache import cache_answer, get_cached_answer
    from notes_chat.retrieval import retrieve_context

    try:
        context_result = retrieve_context(config, question)

        if not context_result.get("success"):
            return generate_conversational_response(
                config,
                f"I couldn't find information about: {question}. Could you try rephrasing your question?",
            )

        context = context_result["context"]
        retrieved_ids = context_result["retrieved_ids"]

        cached_answer = get_cached_answer(question, retrieved_ids)
        if cached_answer:
            return {
                "success": True,
                "answer": cached_answer,
                "sources": context_result.get("sources"),
            }

        fast_prompt = f"""Answer this question about meeting notes: "{question}"

Context: {context}

Answer:"""

        response = requests.post(
            f"{config.models.ollama_url}/api/generate",
            json={
                "model": config.models.llm,
                "prompt": fast_prompt,
                "temperature": 0,
                "top_p": 1,
                "stream": False,
                "options": {"num_predict": 200},  # Limit response length for speed
            },
            timeout=30,
        )

        if response.status_code != 200:
            return {
                "success": False,
                "error": f"Fast search failed: {response.status_code}",
            }

        result = response.json()
        answer = result.get("response", "").strip()

        if answer:
            cache_answer(question, retrieved_ids, answer)
            return {
                "success": True,
                "answer": answer,
                "sources": context_result.get("sources"),
                "search_strategy": "fast search",
            }

        return {"success": False, "error": "Empty response"}

    except Exception as e:  # noqa: BLE001 - fast_search_and_answer: many LLM/retrieval failure modes
        logger.debug("Fast search failed: %s", e)
        return {"success": False, "error": f"Fast search failed: {e}"}


def enhanced_search_and_answer(config: ChirpSettings, question: str) -> dict[str, Any]:
    """Smart routing: fast path for simple cases, full orchestration for complex ones."""
    from notes_chat.cache import cache_answer, get_cached_answer
    from notes_chat.retrieval import retrieve_context

    if is_simple_conversational(question):
        return generate_conversational_response(config, question)

    if is_obvious_search(question):
        return fast_search_and_answer(config, question)

    try:
        orchestration_result = orchestrate_search(config, question)
        if not orchestration_result.get("success"):
            return fast_search_and_answer(config, question)

        search_plan = orchestration_result["search_plan"]

        if not search_plan["requires_search"]:
            return generate_conversational_response(config, question)

        search_terms = " ".join(search_plan["search_terms"])
        time_filter = (
            search_plan["time_filter"] if search_plan["time_filter"] != "null" else None
        )

        context_result = retrieve_context(config, search_terms, time_filter)

        if not context_result.get("success"):
            return fast_search_and_answer(config, question)

        context = context_result["context"]
        retrieved_ids = context_result["retrieved_ids"]

        cached_answer = get_cached_answer(question, retrieved_ids)
        if cached_answer:
            return {
                "success": True,
                "answer": cached_answer,
                "sources": context_result.get("sources"),
            }

        enhanced_prompt = f"""Answer: "{question}"

Context: {context}

Provide a direct answer based on the context."""

        response = requests.post(
            f"{config.models.ollama_url}/api/generate",
            json={
                "model": config.models.llm,
                "prompt": enhanced_prompt,
                "temperature": 0,
                "top_p": 1,
                "stream": False,
                "options": {"num_predict": 300},
            },
            timeout=45,
        )

        if response.status_code != 200:
            return fast_search_and_answer(config, question)

        result = response.json()
        answer = result.get("response", "").strip()

        if answer:
            cache_answer(question, retrieved_ids, answer)
            return {
                "success": True,
                "answer": answer,
                "sources": context_result.get("sources"),
                "search_strategy": search_plan["search_strategy"],
            }

        return fast_search_and_answer(config, question)

    except Exception as exc:  # noqa: BLE001 - enhanced search fallback; many failure modes
        logger.debug("Enhanced search fell back to fast search: %s", exc)
        return fast_search_and_answer(config, question)


def enhanced_search_and_answer_stream(
    config: ChirpSettings,
    question: str,
    client: LLMClient | None = None,
) -> Generator[dict[str, Any], None, None]:
    """Smart routing with streaming: fast path for simple cases, full orchestration for complex ones.

    Grounded answers stream token-by-token through the chirpd daemon
    (``llm.client``). The run-level ``req_id`` is surfaced on the first event
    so the interactive session can cancel an in-flight answer mid-stream.
    """
    from notes_chat.cache import cache_answer, get_cached_answer
    from notes_chat.retrieval import retrieve_context

    req_id = new_request_id()
    yield {"type": "request_started", "req_id": req_id}
    llm = client or LLMClient()

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

    if is_obvious_search(question):
        yield {"type": "thinking", "message": "Searching quickly..."}
        try:
            context_result = retrieve_context(config, question)
            if context_result.get("success"):
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
                    llm, _fast_answer_prompt(question, context), req_id
                ):
                    if event["type"] == "error":
                        yield event
                        return
                    parts.append(event["content"])
                    yield event

                # _stream_answer already emitted an error for an empty stream
                # (and we returned), so any text here is non-empty.
                full_response = "".join(parts)
                cache_answer(question, retrieved_ids, full_response)
                yield {
                    "type": "complete",
                    "answer": full_response,
                    "sources": context_result.get("sources"),
                    "search_strategy": "fast search",
                }
                return
        except Exception as exc:  # noqa: BLE001 - fast-path stream fallback; many failure modes
            logger.debug("Fast-path stream failed, falling through: %s", exc)

    yield {"type": "thinking", "message": "Analyzing question..."}
    try:
        orchestration_result = orchestrate_search(config, question)
        if not orchestration_result.get("success"):
            yield {"type": "thinking", "message": "Falling back to search..."}
            try:
                context_result = retrieve_context(config, question)
                if context_result.get("success"):
                    parts = []
                    for event in _stream_answer(
                        llm,
                        _fast_answer_prompt(question, context_result["context"]),
                        req_id,
                    ):
                        if event["type"] == "error":
                            yield event
                            return
                        parts.append(event["content"])
                        yield event
                    yield {
                        "type": "complete",
                        "answer": "".join(parts),
                        "sources": context_result.get("sources"),
                    }
                else:
                    yield {
                        "type": "error",
                        "message": "Could not find relevant information",
                    }
            except Exception as exc:  # noqa: BLE001 - fallback stream search; many failure modes
                logger.debug("Fallback stream search failed: %s", exc)
                yield {"type": "error", "message": "Search failed"}
            return

        search_plan = orchestration_result["search_plan"]

        if not search_plan["requires_search"]:
            yield {"type": "thinking", "message": "Having a conversation..."}
            parts = []
            for event in _stream_answer(llm, _conversational_prompt(question), req_id):
                if event["type"] == "error":
                    yield event
                    return
                parts.append(event["content"])
                yield event
            yield {"type": "complete", "answer": "".join(parts)}
            return

        search_terms = " ".join(search_plan["search_terms"])
        time_filter = (
            search_plan["time_filter"] if search_plan["time_filter"] != "null" else None
        )

        yield {
            "type": "thinking",
            "message": f"Searching for: {search_plan['search_strategy']}",
        }

        context_result = retrieve_context(config, search_terms, time_filter)
        if not context_result.get("success"):
            yield {
                "type": "thinking",
                "message": "No results found, generating helpful response...",
            }
            fallback_query = f"The user asked: '{question}' but no relevant information was found. Search strategy was: {search_plan['search_strategy']}. Provide a helpful response explaining this and suggest how they might rephrase."
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

        yield {"type": "thinking", "message": "Generating detailed answer..."}
        parts = []
        for event in _stream_answer(
            llm, _grounded_answer_prompt(question, context), req_id
        ):
            if event["type"] == "error":
                yield event
                return
            parts.append(event["content"])
            yield event

        # Empty/whitespace-only streams already yielded an error from
        # _stream_answer (and returned), so any text here is non-empty.
        full_response = "".join(parts)
        cache_answer(question, retrieved_ids, full_response)
        yield {
            "type": "complete",
            "answer": full_response,
            "sources": context_result.get("sources"),
            "search_strategy": search_plan["search_strategy"],
        }

    except Exception as e:  # noqa: BLE001 - enhanced stream search; many failure modes
        logger.debug("Enhanced stream search failed: %s", e)
        yield {"type": "error", "message": f"Enhanced search failed: {e}"}


def is_search_query(question: str) -> bool:
    """Determine if a question requires searching through notes."""
    search_indicators = [
        "what did",
        "who said",
        "when did",
        "what was discussed",
        "tell me about",
        "find",
        "search",
        "look for",
        "show me",
        "what happened",
        "meeting",
        "discussed",
        "action item",
        "decision",
        "summary",
        "topic",
        "agenda",
    ]

    conversational_patterns = [
        "hi",
        "hello",
        "hey",
        "thanks",
        "thank you",
        "how are you",
        "what can you do",
        "what are you",
        "who are you",
        "good morning",
        "good afternoon",
        "goodbye",
        "bye",
        "see you",
    ]

    exact_conversational_patterns = ["help"]

    question_lower = question.lower().strip()

    if question_lower in exact_conversational_patterns:
        return False

    if any(pattern in question_lower for pattern in conversational_patterns):
        return False

    if any(indicator in question_lower for indicator in search_indicators):
        return True

    if len(question.split()) > 5:
        return True

    return False


def validate_ollama_connection(config: ChirpSettings) -> dict[str, Any]:
    """Validate that Ollama is running and the model is available."""
    try:
        response = requests.get(f"{config.models.ollama_url}/api/tags", timeout=5)
        if response.status_code != 200:
            return {"success": False, "error": "Ollama server is not responding"}

        models = response.json().get("models", [])
        model_names = [model["name"] for model in models]

        if config.models.llm not in model_names:
            return {
                "success": False,
                "error": f"Model '{config.models.llm}' not found. Available models: {', '.join(model_names) if model_names else 'none'}. Try: ollama pull {config.models.llm}",
            }

        if config.notes_chat.emb_model not in model_names:
            return {
                "success": False,
                "error": f"Embedding model '{config.notes_chat.emb_model}' not found. Try: ollama pull {config.notes_chat.emb_model}",
            }

        return {"success": True}

    except requests.exceptions.ConnectionError:
        return {
            "success": False,
            "error": "Cannot connect to Ollama. Is it running? Try: ollama serve",
        }
    except Exception as e:  # noqa: BLE001 - fallback after specific request handlers
        logger.debug("Failed to validate Ollama: %s", e)
        return {"success": False, "error": f"Failed to validate Ollama: {e}"}
