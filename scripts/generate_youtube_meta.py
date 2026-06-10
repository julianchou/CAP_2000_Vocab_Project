import argparse
import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

from episode_range_utils import resolve_episode_range
from llm_provider_utils import DEFAULT_NVIDIA_TEXT_MODEL, nvidia_chat_response


load_dotenv()

base_dir = Path(__file__).parent.parent
workspace_dir = Path(os.environ.get("CAP_WORKSPACE_ROOT", str(base_dir / "workspace")))

DEFAULT_LANGUAGE = "zh-TW"
DEFAULT_AUDIO_LANGUAGE = "zh-TW"
DRIVE_LINK_PLACEHOLDER = "{{DRIVE_LINK}}"
GEMINI_MODEL_NAME = os.getenv("CAP_YOUTUBE_META_GEMINI_MODEL", os.getenv("CAP_TEXT_MODEL", "gemini-2.5-pro"))
OPENAI_MODEL_NAME = os.getenv("CAP_YOUTUBE_META_OPENAI_MODEL", os.getenv("OPENAI_TEXT_MODEL", "gpt-4o-mini"))
NVIDIA_MODEL_NAME = os.getenv("CAP_YOUTUBE_META_NVIDIA_MODEL", DEFAULT_NVIDIA_TEXT_MODEL)
VALID_PROVIDERS = {"auto", "gemini", "openai", "nvidia"}

DEFAULT_PROMPT_TEMPLATE = """你是一位熟悉 YouTube SEO 的繁體中文教育頻道企劃。
請根據本集字幕產生 YouTube metadata，主題是會考英文 2000 單字。

要求：
1. title 不要超過 100 個中文字。
2. summary 用 100 個中文字以內摘要本集重點。
3. tags 提供 5 到 10 個繁體中文或英文標籤。
4. words 從字幕整理本集單字時間軸，格式為 time、word、meaning。
5. 單字範圍是 {{START_WORD}} 到 {{END_WORD}}。

字幕：
{{SRT_CONTENT}}

請只輸出 JSON，格式：
{
  "summary": "...",
  "tags": ["..."],
  "words": [
    {"time": "00:00", "word": "...", "meaning": "..."}
  ]
}
"""


def normalize_provider(value: str | None) -> str:
    provider = str(value or os.getenv("CAP_YOUTUBE_META_PROVIDER") or "auto").strip().lower()
    return provider if provider in VALID_PROVIDERS else "auto"


def is_gemini_quota_error(exc: Exception) -> bool:
    text = str(exc)
    return "RESOURCE_EXHAUSTED" in text or "spending cap" in text.lower() or "quota" in text.lower()


def strip_json_fence(text: str) -> str:
    text = str(text or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def extract_json_text(text: str) -> str:
    text = strip_json_fence(text)
    if not text:
        return text
    if text[0] in "[{":
        return text
    match = re.search(r"(\{.*\}|\[.*\])", text, flags=re.DOTALL)
    return match.group(1).strip() if match else text


def generate_meta_json_with_gemini(prompt: str) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured")
    print(f"Calling Gemini to generate YouTube metadata: {GEMINI_MODEL_NAME}")
    response = genai.Client(api_key=api_key).models.generate_content(
        model=GEMINI_MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.2,
        ),
    )
    return getattr(response, "text", "") or ""


def generate_meta_json_with_openai(prompt: str) -> str:
    from openai import OpenAI

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not configured")
    print(f"Calling OpenAI/ChatGPT to generate YouTube metadata: {OPENAI_MODEL_NAME}")
    response = OpenAI().responses.create(
        model=OPENAI_MODEL_NAME,
        input=[
            {
                "role": "system",
                "content": "Return strict JSON only. Do not include Markdown fences or explanations.",
            },
            {"role": "user", "content": prompt},
        ],
    )
    return str(response.output_text or "")


def generate_meta_json_with_nvidia(prompt: str) -> str:
    print(f"Calling NVIDIA to generate YouTube metadata: {NVIDIA_MODEL_NAME}")
    return nvidia_chat_response(
        prompt,
        model=NVIDIA_MODEL_NAME,
        system_prompt="Return strict JSON only. Do not include Markdown fences or explanations.",
        temperature=0.2,
    )


def generate_meta_json(prompt: str, provider: str) -> tuple[dict, str]:
    provider = normalize_provider(provider)
    errors: list[str] = []

    if provider in {"auto", "gemini"}:
        try:
            raw = generate_meta_json_with_gemini(prompt)
            return json.loads(extract_json_text(raw)), f"gemini:{GEMINI_MODEL_NAME}"
        except Exception as exc:
            errors.append(f"Gemini: {exc}")
            if provider == "gemini":
                raise
            if is_gemini_quota_error(exc):
                print("Gemini quota/spending cap reached; falling back to OpenAI/ChatGPT, then NVIDIA if needed.")
            else:
                print(f"Gemini YouTube metadata generation failed; falling back to OpenAI/ChatGPT, then NVIDIA if needed: {exc}")

    if provider in {"auto", "openai"}:
        try:
            raw = generate_meta_json_with_openai(prompt)
            return json.loads(extract_json_text(raw)), f"openai:{OPENAI_MODEL_NAME}"
        except Exception as exc:
            errors.append(f"OpenAI: {exc}")
            if provider == "openai":
                raise RuntimeError("YouTube metadata generation failed: " + " | ".join(errors)) from exc
            print(f"OpenAI YouTube metadata generation failed; falling back to NVIDIA: {exc}")

    if provider in {"auto", "nvidia"}:
        try:
            raw = generate_meta_json_with_nvidia(prompt)
            return json.loads(extract_json_text(raw)), f"nvidia:{NVIDIA_MODEL_NAME}"
        except Exception as exc:
            errors.append(f"NVIDIA: {exc}")
            raise RuntimeError("YouTube metadata generation failed: " + " | ".join(errors)) from exc

    raise RuntimeError("YouTube metadata generation failed: " + " | ".join(errors))


def detect_profile_id() -> str:
    profile_id = str(os.environ.get("CAP_PROFILE_ID", "")).strip()
    if profile_id:
        return profile_id
    if workspace_dir.parent.name == "workspaces" and workspace_dir.name:
        return workspace_dir.name
    return "default"


def shared_prompt_path() -> Path:
    return base_dir / "config" / detect_profile_id() / "prompts" / "youtube_meta_prompt.txt"


def ensure_prompt_template(_target_folder: Path) -> Path:
    prompt_path = shared_prompt_path()
    if not prompt_path.exists():
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_path.write_text(DEFAULT_PROMPT_TEMPLATE, encoding="utf-8")
    return prompt_path


def render_prompt(template_text: str, start_word: int, end_word: int, srt_content: str) -> str:
    return (
        template_text
        .replace("{{START_WORD}}", str(start_word))
        .replace("{{END_WORD}}", str(end_word))
        .replace("{{SRT_CONTENT}}", srt_content)
    )


def build_drive_section(drive_link: str = DRIVE_LINK_PLACEHOLDER) -> str:
    return f"📥 本集單字表下載 (Google Drive)：\n{drive_link}"


def build_final_description(ai_data: dict, drive_link: str = DRIVE_LINK_PLACEHOLDER) -> str:
    words_list = ai_data.get("words", [])
    timeline_text = "\n".join(
        f"{item.get('time', '00:00')} {item.get('word', '')}：{item.get('meaning', '')}"
        for item in words_list
    )

    return f"""📌 本集重點：
{ai_data.get("summary", "")}

📥 下載本集單字表：
{build_drive_section(drive_link)}

⏱ 單字時間軸：
{timeline_text}

#會考英文 #英文單字 #2000單字 #國中英文 #英文聽力"""


def generate_youtube_meta(ep_num: int, provider: str = "auto"):
    print(f"\n[YouTube Meta] episode={ep_num:02d} generating metadata...")
    target_folder, start_word, end_word = resolve_episode_range(workspace_dir, ep_num)

    srt_path = target_folder / "02_subtitles" / "notebooklm_audio_fixed.srt"
    output_folder = target_folder / "05_output"
    meta_output_path = output_folder / "youtube_meta.json"

    if not srt_path.exists():
        print(f"subtitle file not found: {srt_path}")
        return

    srt_content = srt_path.read_text(encoding="utf-8")
    prompt_template_path = ensure_prompt_template(target_folder)
    prompt_template = prompt_template_path.read_text(encoding="utf-8")
    prompt = render_prompt(prompt_template, start_word, end_word, srt_content)

    provider = normalize_provider(provider)
    print(
        f"youtube_meta_provider={provider} "
        f"gemini_model={GEMINI_MODEL_NAME} openai_model={OPENAI_MODEL_NAME} "
        f"nvidia_model={NVIDIA_MODEL_NAME}"
    )

    try:
        ai_data, provider_used = generate_meta_json(prompt, provider)
        print(f"YouTube metadata provider used: {provider_used}")
        final_description = build_final_description(ai_data)

        fixed_title = (
            f"考前必聽｜國中英文會考2000單字攻略-EP{ep_num:02d} "
            f"({start_word:03d}-{end_word:03d})"
        )

        meta_data = {
            "title": fixed_title,
            "description": final_description,
            "tags": ai_data.get("tags", []),
            "drive_link": "",
            "drive_link_placeholder": DRIVE_LINK_PLACEHOLDER,
            "default_language": DEFAULT_LANGUAGE,
            "default_audio_language": DEFAULT_AUDIO_LANGUAGE,
        }

        output_folder.mkdir(parents=True, exist_ok=True)
        meta_output_path.write_text(
            json.dumps(meta_data, ensure_ascii=False, indent=4),
            encoding="utf-8",
        )
        print(f"YouTube Metadata saved to: {meta_output_path}")
    except Exception as exc:
        print(f"YouTube metadata generation failed: {exc}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--provider",
        choices=sorted(VALID_PROVIDERS),
        default=os.getenv("CAP_YOUTUBE_META_PROVIDER", "auto"),
        help="LLM provider for YouTube metadata. auto tries Gemini first, then OpenAI, then NVIDIA.",
    )
    parser.add_argument("--ep", type=int, help="集數")
    args = parser.parse_args()

    if args.ep:
        generate_youtube_meta(args.ep, args.provider)
    else:
        print("請指定 --ep，例如 --ep 3")


if __name__ == "__main__":
    main()
