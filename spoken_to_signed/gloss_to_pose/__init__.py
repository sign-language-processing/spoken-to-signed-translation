from typing import Union

from pose_format import Pose

from ..text_to_gloss.types import Gloss
from .concatenate import concatenate_poses
from .lookup import CSVPoseLookup, PoseLookup, PoseResult  # noqa: F401


def gloss_to_pose(
    glosses: Gloss,
    pose_lookup: PoseLookup,
    spoken_language: str,
    signed_language: str,
    source: str = None,
    anonymize: Union[bool, Pose] = False,
) -> PoseResult:
    results = pose_lookup.lookup_sequence(glosses, spoken_language, signed_language, source)
    poses = [r.pose for r in results]

    if anonymize:
        try:
            from pose_anonymization.appearance import (
                remove_appearance,
                transfer_appearance,
            )
        except ImportError as e:
            raise ImportError(
                "Please install pose_anonymization. "
                "pip install git+https://github.com/sign-language-processing/pose-anonymization"
            ) from e

        if isinstance(anonymize, Pose):
            print("Transferring appearance...")
            poses = [transfer_appearance(pose, anonymize) for pose in poses]
        else:
            print("Removing appearance...")
            poses = [remove_appearance(pose) for pose in poses]

    return PoseResult(pose=concatenate_poses(poses))
