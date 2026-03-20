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


def _run_text_to_gloss_to_pose(tmp_dir: str, text: str, extra_args: list[str] = None):
    pose_path = Path(tmp_dir) / "output.pose"

    cmd = [
        "text_to_gloss_to_pose",
        "--text", text,
        "--glosser", "simple",
        "--lexicon", "assets/dummy_lexicon",
        "--spoken-language", "de",
        "--signed-language", "sgg",
        "--pose", str(pose_path),
    ]
    if extra_args:
        cmd.extend(extra_args)

    result = subprocess.run(cmd, capture_output=True, text=True)
    return result, pose_path


def test_fingerspelling_produces_pose():
    with tempfile.TemporaryDirectory() as tmp_dir:
        result, pose_path = _run_text_to_gloss_to_pose(tmp_dir, "abcd")
        assert result.returncode == 0, f"Command failed: {result.stderr}"
        assert pose_path.exists(), "Pose file was not created"

        with open(pose_path, "rb") as f:
            pose = Pose.read(f.read())

        assert pose.body.data.shape[0] > 0, "Fingerspelled word should produce frames"


def test_no_fingerspelling_fails_for_unknown_word():
    with tempfile.TemporaryDirectory() as tmp_dir:
        result, _ = _run_text_to_gloss_to_pose(tmp_dir, "abcd", ["--disable-fingerspelling"])
        assert result.returncode != 0, "Should fail without fingerspelling for unknown word"