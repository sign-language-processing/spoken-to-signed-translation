from dataclasses import dataclass, field
from typing import Optional, Union

from pose_format import Pose

from ..text_to_gloss.types import Gloss
from .concatenate import concatenate_poses
from .coverage import TokenCoverage
from .lookup import CSVPoseLookup as CSVPoseLookup
from .lookup import PoseLookup


@dataclass
class GlossToPoseResult:
    """Return type of gloss_to_pose(). Always contains a pose; token_coverages
    is populated only when coverage_info=True is passed."""

    pose: Pose
    token_coverages: Optional[list[TokenCoverage]] = field(default=None)


def gloss_to_pose(
    glosses: Gloss,
    pose_lookup: PoseLookup,
    spoken_language: str,
    signed_language: str,
    source: str = None,
    anonymize: Union[bool, Pose] = False,
    coverage_info: bool = False,
) -> GlossToPoseResult:
    # Transform the list of glosses into a list of poses
    result = pose_lookup.lookup_sequence(glosses, spoken_language, signed_language, source, coverage_info=coverage_info)
    if coverage_info:
        poses, token_coverages = result
    else:
        poses = result
        token_coverages = None

    # Anonymize poses
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

    # Concatenate the poses to create a single pose
    pose = concatenate_poses(poses)

    return GlossToPoseResult(pose=pose, token_coverages=token_coverages)
