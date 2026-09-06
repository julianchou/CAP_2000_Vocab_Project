import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


MODULE_PATH = Path(__file__).resolve().parent.parent / "scripts" / "jimeng_cli.py"
SPEC = importlib.util.spec_from_file_location("jimeng_cli", MODULE_PATH)
jimeng_cli = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = jimeng_cli
SPEC.loader.exec_module(jimeng_cli)


class JimengCliTests(unittest.TestCase):
    def test_build_prompt_includes_video_options(self):
        prompt = jimeng_cli.build_prompt(
            "John waves gently.",
            resolution="720p",
            duration=5,
            aspect_ratio="16:9",
            camera_fixed=True,
            watermark=False,
        )

        self.assertTrue(prompt.startswith("John waves gently."))
        self.assertIn("--resolution 720p", prompt)
        self.assertIn("--duration 5", prompt)
        self.assertIn("--ratio 16:9", prompt)
        self.assertIn("--camerafixed true", prompt)
        self.assertIn("--watermark false", prompt)

    def test_image_reference_encodes_local_file(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "sample.png"
            image.write_bytes(b"\x89PNG\r\n")

            reference = jimeng_cli.image_reference(str(image))

        self.assertTrue(reference.startswith("data:image/png;base64,"))

    def test_video_url_supports_content_object(self):
        payload = {
            "status": "succeeded",
            "content": {"video_url": "https://example/video.mp4"},
        }

        self.assertEqual(
            jimeng_cli.video_url(payload),
            "https://example/video.mp4",
        )

    def test_create_task_builds_expected_request(self):
        captured = {}

        def fake_request_json(method, url, **kwargs):
            captured.update(method=method, url=url, kwargs=kwargs)
            return {"id": "task-123", "status": "queued"}

        env = {
            **os.environ,
            "ARK_API_KEY": "test-key",
            "ARK_BASE_URL": "https://ark.example/api/v3",
        }
        with patch.dict(os.environ, env, clear=True), patch.object(
            jimeng_cli,
            "request_json",
            side_effect=fake_request_json,
        ):
            result = jimeng_cli.create_task(
                model="video-model",
                prompt="A subtle camera push-in.",
                image="https://example/image.png",
            )

        self.assertEqual(result["id"], "task-123")
        self.assertEqual(captured["method"], "POST")
        self.assertEqual(
            captured["url"],
            "https://ark.example/api/v3/contents/generations/tasks",
        )
        self.assertEqual(captured["kwargs"]["json"]["model"], "video-model")
        self.assertEqual(
            captured["kwargs"]["json"]["content"][1]["image_url"]["url"],
            "https://example/image.png",
        )


if __name__ == "__main__":
    unittest.main()
