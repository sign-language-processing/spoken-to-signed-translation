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
