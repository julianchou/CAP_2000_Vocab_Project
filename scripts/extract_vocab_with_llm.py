import argparse
import re
from pathlib import Path

import pandas as pd
from PyPDF2 import PdfReader


SOURCE_POS_VALUES = (
    "\u6240\u6709\u683c",
    "\u9650\u5b9a\u8a5e",
    "\u4ee3\u540d\u8a5e",
    "\u5f62\u5bb9\u8a5e",
    "\u9023\u63a5\u8a5e",
    "\u4ecb\u7cfb\u8a5e",
    "\u52a9\u52d5\u8a5e",
    "\u611f\u5606\u8a5e",
    "\u7591\u554f\u8a5e",
    "\u526f\u8a5e",
    "\u52d5\u8a5e",
    "\u540d\u8a5e",
    "\u6578\u8a5e",
)
POS_PATTERN = "|".join(re.escape(value) for value in SOURCE_POS_VALUES)
COMPOUND_POS_PATTERN = rf"(?:{POS_PATTERN})(?:\s*/\s*(?:{POS_PATTERN}))*"
EXPECTED_ENTRY_COUNT = 2000


def normalize_pdf_text(text: str) -> str:
    normalized = " ".join(str(text or "").split())
    for pos in SOURCE_POS_VALUES:
        split_pos_pattern = r"\s*".join(re.escape(character) for character in pos)
        normalized = re.sub(split_pos_pattern, pos, normalized)
    return normalized


def normalize_meaning(text: str) -> str:
    # PDF line wrapping inserts spaces inside Chinese words, such as "因 為".
    return re.sub(r"\s+", "", str(text or "")).strip()


def load_existing_words(index_path: Path) -> list[dict[str, object]]:
    if not index_path.exists():
        raise FileNotFoundError(
            f"Existing word index is required to identify exact entry boundaries: {index_path}"
        )

    frame = pd.read_csv(index_path, usecols=["Index", "Word"])
    frame["Index"] = pd.to_numeric(frame["Index"], errors="raise").astype(int)
    frame["Word"] = frame["Word"].fillna("").astype(str).str.strip()
    frame = frame.sort_values("Index").drop_duplicates(subset=["Index"], keep="first")

    expected = list(range(1, EXPECTED_ENTRY_COUNT + 1))
    actual = frame["Index"].tolist()
    if actual != expected:
        raise ValueError(
            f"Existing index must contain contiguous entries 1-{EXPECTED_ENTRY_COUNT}; "
            f"found {len(actual)} entries."
        )
    if frame["Word"].eq("").any():
        raise ValueError("Existing index contains blank Word values.")
    return frame[["Index", "Word"]].to_dict(orient="records")


def find_entry(
    pages: list[str],
    index: int,
    word: str,
    next_index: int | None,
    next_word: str | None,
) -> dict[str, object]:
    start_pattern = re.compile(
        rf"(?<!\d){index}\s+{re.escape(word)}\s+(?P<pos>{COMPOUND_POS_PATTERN})\s+"
    )
    matches: list[tuple[int, re.Match[str]]] = []
    for page_number, page_text in enumerate(pages, start=1):
        match = start_pattern.search(page_text)
        if match:
            matches.append((page_number, match))

    if len(matches) != 1:
        raise ValueError(
            f"Entry {index} {word!r}: expected one PDF match, found {len(matches)}."
        )

    page_number, match = matches[0]
    page_text = pages[page_number - 1]
    end = len(page_text)
    if next_index is not None and next_word is not None:
        next_pattern = re.compile(
            rf"(?<!\d){next_index}\s+{re.escape(next_word)}\s+{COMPOUND_POS_PATTERN}\s+"
        )
        next_match = next_pattern.search(page_text, match.end())
        if next_match:
            end = next_match.start()

    meaning = normalize_meaning(page_text[match.end():end])
    if not meaning:
        raise ValueError(f"Entry {index} {word!r}: blank meaning extracted from page {page_number}.")

    return {
        "Index": index,
        "Word": word,
        "POS": re.sub(r"\s+", "", match.group("pos")),
        "Meaning": meaning,
    }


def extract_vocab_index(pdf_path: Path, index_path: Path) -> pd.DataFrame:
    seed_rows = load_existing_words(index_path)
    reader = PdfReader(str(pdf_path))
    pages = [normalize_pdf_text(page.extract_text() or "") for page in reader.pages]

    rows: list[dict[str, object]] = []
    for offset, seed in enumerate(seed_rows):
        next_seed = seed_rows[offset + 1] if offset + 1 < len(seed_rows) else None
        rows.append(
            find_entry(
                pages=pages,
                index=int(seed["Index"]),
                word=str(seed["Word"]),
                next_index=int(next_seed["Index"]) if next_seed else None,
                next_word=str(next_seed["Word"]) if next_seed else None,
            )
        )

    frame = pd.DataFrame(rows, columns=["Index", "Word", "POS", "Meaning"])
    if len(frame) != EXPECTED_ENTRY_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_ENTRY_COUNT} extracted entries, found {len(frame)}."
        )
    if frame[["Word", "POS", "Meaning"]].replace("", pd.NA).isna().any().any():
        raise ValueError("Extracted index contains blank required fields.")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild vocab_index.csv from the source PDF with authoritative POS and meaning."
    )
    parser.add_argument("--pdf", type=Path, help="Override source PDF path.")
    parser.add_argument("--output", type=Path, help="Override vocab index output path.")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent.parent
    pdf_path = args.pdf or base_dir / "core" / "assets" / "cap_vocab_2000_source.pdf"
    output_path = args.output or base_dir / "core" / "assets" / "vocab_index.csv"
    if not pdf_path.exists():
        raise FileNotFoundError(f"Source PDF not found: {pdf_path}")

    frame = extract_vocab_index(pdf_path, output_path)
    frame.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Wrote {len(frame)} authoritative entries to {output_path}", flush=True)


if __name__ == "__main__":
    main()
