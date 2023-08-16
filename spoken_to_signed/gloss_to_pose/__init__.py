
from pose_format import Pose

from ..text_to_gloss.types import Gloss
from .concatenate import concatenate_poses
from .lookup import PoseLookup, CSVPoseLookup


def gloss_to_pose(glosses: Gloss, pose_lookup: PoseLookup, spoken_language: str, signed_language: str) -> Pose:
    # Transform the list of glosses into a list of poses
    poses = pose_lookup.lookup_sequence(glosses, spoken_language, signed_language)

    # Concatenate the poses to create a single pose
    return concatenate_poses(poses)
