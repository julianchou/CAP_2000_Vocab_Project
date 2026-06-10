import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print("[RUN] " + " ".join(cmd), flush=True)
    result = subprocess.run(cmd, check=False, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"command failed return_code={result.returncode}: {' '.join(cmd)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Short generator: regenerate scene image, animation prompt, then animation.")
    parser.add_argument("--ep-dir", required=True)
    parser.add_argument("--storyboard", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--image-provider", default="gemini", choices=["gemini", "openai", "nvidia"])
    parser.add_argument("--animation-provider", default="gemini", choices=["gemini", "openai", "nvidia"])
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    run([
        sys.executable,
        "-u",
        str(root / "short_generate_storyboard_images.py"),
        "--ep-dir",
        args.ep_dir,
        "--storyboard",
        args.storyboard,
        "--manifest",
        args.manifest,
        "--force",
        "--scene-id",
        args.scene_id,
        "--provider",
        args.image_provider,
    ])
    run([
        sys.executable,
        "-u",
        str(root / "short_generate_animation_prompt.py"),
        "--storyboard",
        args.storyboard,
        "--scene-id",
        args.scene_id,
        "--provider",
        args.animation_provider,
    ])
    run([
        sys.executable,
        "-u",
        str(root / "short_generate_animation.py"),
        "--ep-dir",
        args.ep_dir,
        "--storyboard",
        args.storyboard,
        "--scene-id",
        args.scene_id,
    ])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
