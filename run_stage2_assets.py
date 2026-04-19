import argparse
import subprocess
import sys
from pathlib import Path


def run_step(script_path: Path, *args: str):
    subprocess.run([sys.executable, str(script_path), *args], check=False)


def main():
    parser = argparse.ArgumentParser(description="批次產生素材與克漏字資產")
    parser.add_argument("--ep", type=int, help="處理單一集數")
    parser.add_argument("--start", type=int, help="起始集數")
    parser.add_argument("--end", type=int, help="結束集數")
    args = parser.parse_args()

    if args.ep is not None:
        start_ep = args.ep
        end_ep = args.ep
    elif args.start is not None:
        start_ep = args.start
        end_ep = args.end if args.end is not None else args.start
    else:
        parser.error("請提供 --ep，或提供 --start 與可選的 --end。")

    scripts_dir = Path(__file__).resolve().parent / "scripts"

    for ep in range(start_ep, end_ep + 1):
        print(f"\n🚀 --- 開始處理第 {ep:02d} 集素材生成 ---")
        run_step(scripts_dir / "generate_vocab_content.py", "--ep", str(ep))
        run_step(scripts_dir / "generate_flashcards.py", "--ep", str(ep))
        run_step(scripts_dir / "generate_cloze_quiz.py", "--ep", str(ep))
        run_step(scripts_dir / "generate_cloze_flashcards.py", "--ep", str(ep))
        run_step(scripts_dir / "generate_cover.py", "--ep", str(ep))
        run_step(scripts_dir / "generate_notebooklm_prompt.py", "--ep", str(ep))

    print(f"\n✅ 階段 2 完成：第 {start_ep:02d} 至 {end_ep:02d} 集素材已備妥。")


if __name__ == "__main__":
    main()
