from typing import Any

import requests

from config.settings import ChirpSettings

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


def generate_answer(
    config: ChirpSettings, question: str, context: str
) -> dict[str, Any]:
    """Generate an answer using Ollama LLM with deterministic settings."""
    try:
        if not context.strip():
            return {"success": False, "error": "Empty context provided"}

        prompt = SYSTEM_PROMPT.format(context=context, question=question)

        response = requests.post(
            f"{config.models.ollama_url}/api/generate",
            json={
                "model": config.models.llm,
                "prompt": prompt,
                "temperature": 0,
                "top_p": 1,
                "stream": False,
            },
            timeout=60,
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

        no_info_patterns = [
            "i don't have enough information",
            "not enough information",
            "cannot answer",
            "unable to answer",
            "insufficient information",
        ]

        if any(pattern in answer.lower() for pattern in no_info_patterns):
            return {
                "success": False,
                "error": "No relevant information found in the context",
                "answer": answer,
            }

        return {"success": True, "answer": answer}

    except requests.exceptions.ConnectionError:
        return {
            "success": False,
            "error": "Cannot connect to Ollama. Is it running? Try: ollama serve",
        }
    except requests.exceptions.Timeout:
        return {
            "success": False,
            "error": "Ollama request timed out. The model might be too large or the query too complex.",
        }
    except ConnectionError:
        return {
            "success": False,
            "error": "Cannot connect to Ollama. Is it running? Try: ollama serve",
        }
    except Exception as e:
        return {"success": False, "error": f"Failed to generate answer: {e}"}


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
    except Exception as e:
        return {"success": False, "error": f"Failed to validate Ollama: {e}"}
