import json
import subprocess
import tempfile
from pathlib import Path

from pose_format import Pose


def test_text_to_gloss_to_pose():
    with tempfile.TemporaryDirectory() as tmp_dir:
        pose_path = Path(tmp_dir) / "output.pose"

        result = subprocess.run(
            [
                "text_to_gloss_to_pose",
                "--text", "Kleine Kinder essen Pizza in Zürich.",
                "--glosser", "simple",
                "--lexicon", "assets/dummy_lexicon",
                "--spoken-language", "de",
                "--signed-language", "sgg",
                "--pose", str(pose_path),
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, f"Command failed: {result.stderr}"
        assert pose_path.exists(), "Pose file was not created"

        with open(pose_path, "rb") as f:
            pose = Pose.read(f.read())

        assert pose.body.data.shape[0] > 0, "Pose has no frames"


def test_text_to_gloss_to_pose_coverage_info():
    with tempfile.TemporaryDirectory() as tmp_dir:
        pose_path = Path(tmp_dir) / "output.pose"

        result = subprocess.run(
            [
                "text_to_gloss_to_pose",
                "--text", "Kleine Kinder essen Pizza in Zürich.",
                "--glosser", "simple",
                "--lexicon", "assets/dummy_lexicon",
                "--spoken-language", "de",
                "--signed-language", "sgg",
                "--pose", str(pose_path),
                "--coverage-info",
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, f"Command failed: {result.stderr}"
        assert "Coverage:" in result.stdout, "Coverage summary not printed"


def test_text_to_gloss_to_pose_coverage_stats():
    with tempfile.TemporaryDirectory() as tmp_dir:
        pose_path = Path(tmp_dir) / "output.pose"
        stats_path = Path(tmp_dir) / "coverage.json"

        result = subprocess.run(
            [
                "text_to_gloss_to_pose",
                "--text", "Kleine Kinder essen Pizza in Zürich.",
                "--glosser", "simple",
                "--lexicon", "assets/dummy_lexicon",
                "--spoken-language", "de",
                "--signed-language", "sgg",
                "--pose", str(pose_path),
                "--coverage-stats", str(stats_path),
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, f"Command failed: {result.stderr}"
        assert stats_path.exists(), "Coverage stats file was not created"

        with open(stats_path, encoding="utf-8") as f:
            data = json.load(f)

        assert "coverage" in data
        assert "total_tokens" in data
        assert "matched_tokens" in data
        assert "sentences" in data
        assert len(data["sentences"]) == 1
        assert len(data["sentences"][0]["tokens"]) > 0
