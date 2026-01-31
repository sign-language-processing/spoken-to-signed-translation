from typing import Union, Tuple
from pose_format import Pose

from .concatenate import concatenate_poses
from .lookup import PoseLookup, CSVPoseLookup
from ..text_to_gloss.types import Gloss

def gloss_to_pose(
    glosses: Gloss,
    pose_lookup: PoseLookup,
    spoken_language: str,
    signed_language: str,
    source: str = None,
    anonymize: Union[bool, Pose] = False,
    coverage_info: bool = False,
) -> Union[Pose, Tuple[Pose, str]]:
    """
    Transform a sequence of glosses into a Pose.
    
    If coverage_info=True, also returns a coverage string.
    """

    # ---- Lookup ----
    if coverage_info:
        poses, coverage = pose_lookup.lookup_sequence(
            glosses,
            spoken_language,
            signed_language,
            source,
            coverage_info=True,
        )
    else:
        poses = pose_lookup.lookup_sequence(
            glosses,
            spoken_language,
            signed_language,
            source,
        )

    # ---- Anonymization (unchanged) ----
    if anonymize:
        try:
            from pose_anonymization.appearance import remove_appearance, transfer_appearance
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

    # ---- Concatenate ----
    pose = concatenate_poses(poses)

    if coverage_info:
        return pose, coverage

    return pose