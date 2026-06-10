import argparse
import sys
import traceback

from gen_matched_preview_v4 import generate_preview
from generate_scene_animation import SceneAnimationError, generate_scene_animation


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate one scene image and then generate its animation")
    parser.add_argument("--ep", type=int, required=True)
    parser.add_argument("--scene_id", required=True)
    parser.add_argument("--provider", default="auto", choices=["auto", "gemini", "openai", "nvidia"], help="Image provider")
    parser.add_argument("--animation_provider", default="gemini", choices=["gemini", "openai"])
    args = parser.parse_args()

    scene_id = str(args.scene_id).strip()
    scene_id_int = int(scene_id)

    print("STEP A/2 regenerate scene image")
    print(f"episode={args.ep}")
    print(f"scene_id={scene_id}")
    generated_image_path = generate_preview(args.ep, scene_id=scene_id_int, provider=args.provider)
    if not generated_image_path:
        raise FileNotFoundError(f"failed to regenerate image for scene_id={scene_id}")
    print(f"generated_image_path={generated_image_path}")

    print("STEP B/2 generate scene animation")
    generate_scene_animation(
        args.ep,
        scene_id,
        image_path_override=str(generated_image_path),
        provider=args.animation_provider,
    )
    print("DONE scene image and animation generated")


if __name__ == "__main__":
    try:
        main()
    except (SceneAnimationError, FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"FATAL: {e}")
        traceback.print_exc()
        sys.exit(1)
