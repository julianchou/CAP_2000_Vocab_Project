import os
from typing import Any


NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
DEFAULT_NVIDIA_TEXT_MODEL = os.getenv(
    "CAP_NVIDIA_TEXT_MODEL",
    os.getenv("NVIDIA_TEXT_MODEL", "nvidia/nemotron-3-super-120b-a12b"),
)


def nvidia_chat_response(
    prompt: str,
    *,
    model: str | None = None,
    system_prompt: str = "Return only the requested content.",
    temperature: float = 0.2,
) -> str:
    from openai import OpenAI

    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        raise RuntimeError("NVIDIA_API_KEY is not configured")

    client = OpenAI(api_key=api_key, base_url=NVIDIA_BASE_URL)
    selected_model = model or DEFAULT_NVIDIA_TEXT_MODEL
    try:
        response = client.chat.completions.create(
            model=selected_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
        )
    except Exception as exc:
        message = str(exc)
        if "404" in message or "Not Found" in message:
            raise RuntimeError(
                f"NVIDIA model not found or unavailable: {selected_model}. "
                "Update NVIDIA_TEXT_MODEL or CAP_NVIDIA_TEXT_MODEL to a model listed on build.nvidia.com."
            ) from exc
        if "504" in message:
            raise RuntimeError(
                f"NVIDIA hosted endpoint timed out for model: {selected_model}. "
                "Use a smaller model for this step or retry later."
            ) from exc
        raise
    choice = response.choices[0] if response.choices else None
    message = getattr(choice, "message", None)
    content = getattr(message, "content", None)
    if content:
        return str(content)

    finish_reason = getattr(choice, "finish_reason", "") if choice is not None else ""
    response_data: dict[str, Any] = {}
    if hasattr(response, "model_dump"):
        response_data = response.model_dump()
    elif hasattr(response, "dict"):
        response_data = response.dict()
    preview = str(response_data or response)[:1000]
    raise RuntimeError(
        f"NVIDIA returned an empty chat message for model {selected_model}; "
        f"finish_reason={finish_reason or 'unknown'} response_preview={preview}"
    )
