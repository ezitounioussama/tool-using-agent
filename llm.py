"""Minimal Ollama client. Standard library only -- no packages to install."""

import json
import urllib.error
import urllib.request

HOST = "http://localhost:11434"
MODEL = "llama3.2:3b"

# Kept low so the same question gives the same answer twice; the point of this
# checkpoint is the routing, and a wandering model makes routes hard to compare.
TEMPERATURE = 0.1


class LLMError(RuntimeError):
    """The model could not be reached or refused to answer."""


def _post(path, payload, timeout):
    request = urllib.request.Request(
        HOST + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def generate(prompt, system=None, timeout=60, max_tokens=220):
    """Ask the model one question and return its answer as text."""
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": TEMPERATURE, "num_predict": max_tokens},
    }
    if system:
        payload["system"] = system

    try:
        data = _post("/api/generate", payload, timeout)
    except urllib.error.URLError as error:
        raise LLMError(f"cannot reach Ollama at {HOST}: {error}") from error
    except TimeoutError as error:
        raise LLMError(f"Ollama did not answer within {timeout}s") from error

    answer = (data.get("response") or "").strip()
    if not answer:
        raise LLMError("Ollama returned an empty response")
    return answer


def health():
    """Return (ok, message). Used by the runners to fail with instructions."""
    try:
        with urllib.request.urlopen(HOST + "/api/tags", timeout=4) as response:
            tags = json.loads(response.read())
    except Exception as error:  # noqa: BLE001 - any failure means "not usable"
        return False, f"Ollama is not reachable at {HOST} ({error}). Start it with: ollama serve"

    installed = [model["name"] for model in tags.get("models", [])]
    if not any(name.startswith(MODEL.split(":")[0]) for name in installed):
        return False, f"Model {MODEL} is not installed. Install it with: ollama pull {MODEL}"
    return True, f"Ollama up, {MODEL} available"
