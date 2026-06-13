import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types

import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
if "dotenv" not in sys.modules:
    dotenv_stub = types.ModuleType("dotenv")
    dotenv_stub.load_dotenv = lambda: None
    sys.modules["dotenv"] = dotenv_stub


def load_script(name: str):
    path = SCRIPTS_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_reconcile_batch_preserves_phonetic():
    module = load_script("generate_vocab_content")
    source = [{"Index": 1, "Word": "a few", "POS": "限定詞", "Meaning": "一些"}]
    generated = [{
        "Index": "1",
        "Word": "a few",
        "Phonetic": "/ə fjuː/",
        "English_Sentence": "I have a few books.",
        "Chinese_Translation": "我有幾本書。",
    }]

    result = module.reconcile_batch(source, generated)

    assert result[0]["Phonetic"] == "/ə fjuː/"
    assert result[0]["POS"] == "det."


def test_render_flashcard_accepts_phonetic_and_legacy_rows():
    module = load_script("generate_flashcards")
    base = Image.new("RGB", (1920, 1080), "#204060")
    common = {
        "Word": "example",
        "POS": "n.",
        "Meaning": "例子",
        "English_Sentence": "This is an example.",
        "Chinese_Translation": "這是一個例子。",
    }

    with_phonetic = module.render_flashcard(
        base, pd.Series({**common, "Phonetic": "/ɪɡˈzæmpəl/"}), None
    )
    legacy = module.render_flashcard(base, pd.Series(common), None)

    assert with_phonetic.size == base.size
    assert legacy.size == base.size


def test_no3_outputs_include_phonetic_in_csv_and_json():
    module = load_script("generate_vocab_content")
    rows = [{
        "Index": 1,
        "Word": "example",
        "Phonetic": "/ɪɡˈzæmpəl/",
        "POS": "n.",
        "Meaning": "例子",
        "English_Sentence": "This is an example.",
        "Chinese_Translation": "這是一個例子。",
    }]

    with tempfile.TemporaryDirectory() as temp_dir:
        csv_path, json_path = module.write_vocab_outputs(rows, Path(temp_dir))
        csv_df = pd.read_csv(csv_path)
        json_rows = json.loads(json_path.read_text(encoding="utf-8"))

    assert "Phonetic" in csv_df.columns
    assert csv_df.loc[0, "Phonetic"] == "/ɪɡˈzæmpəl/"
    assert json_rows[0]["Phonetic"] == "/ɪɡˈzæmpəl/"
