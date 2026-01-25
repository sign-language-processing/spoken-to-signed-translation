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

    # Lookup
    result = pose_lookup.lookup_sequence(
        glosses,
        spoken_language,
        signed_language,
        source,
        coverage_info=coverage_info,
    )
    if coverage_info:
        poses, coverage = result
    else:
        poses = result


    # Anonymize poses
    if anonymize:
        try:
            from pose_anonymization.appearance import remove_appearance, transfer_appearance
        except ImportError as e:
            raise ImportError("Please install pose_anonymization. "
                              "pip install git+https://github.com/sign-language-processing/pose-anonymization") from e

        if isinstance(anonymize, Pose):
            print("Transferring appearance...")
            poses = [transfer_appearance(pose, anonymize) for pose in poses]
        else:
            print("Removing appearance...")
            poses = [remove_appearance(pose) for pose in poses]

    # Concatenate the poses to create a single pose
    pose = concatenate_poses(poses)

    return (pose, coverage) if coverage_info else pose
