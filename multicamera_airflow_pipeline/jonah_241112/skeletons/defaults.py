import sys
import numpy as np

# load skeleton
try:
    from multicamera_airflow_pipeline.jonah_241112.skeletons.weinreb15pt import (
        dataset_info,
        conf_thresholds_by_camera,
        parents_dict,
        front_keypoints,
        side_keypoints,
        back_keypoints,
    )
except:

    # Add the directory containing the file to the system path
    sys.path.append(
        "/n/groups/datta/Jonah/Local_code_groups/6cam_repos/multicam_airflow_pipeline/multicamera_airflow_pipeline/jonah_241112/skeletons/weinreb15pt.py"
    )
    # Now import the dataset_info dictionary
    from weinreb15pt import dataset_info, conf_thresholds_by_camera, parents_dict
keypoint_info = dataset_info["keypoint_info"]
skeleton_info = dataset_info["skeleton_info"]
conf_thresholds_by_camera = conf_thresholds_by_camera
keypoints = [keypoint_info[i]["name"] for i in keypoint_info.keys()]
keypoints = np.array(keypoints)
keypoints_order = keypoints
kpt_dict = {j: i for i, j in enumerate(keypoints_order)}

# all mice will be projected into this template
# lengths are from the keypoint to its parent in the hierarchy below
# where: parent = hierarchy[keypoint][0]
default_template_bone_length_mean = {
    "spine_low": 23.7,
    "tail_base": 12.8,
    "spine_mid": 11.55,
    "spine_high": 11.55,
    "left_ear": 19.8,
    "right_ear": 19.8,
    "forehead": 20.4,
    "nose_tip": 15.1,
    "left_fore_paw": 34.2,
    "right_fore_paw": 34.2,
    "left_hind_paw_back": 25.6,
    "left_hind_paw_front": 11.2,
    "right_hind_paw_back": 25.6,
    "right_hind_paw_front": 11.2,
}

default_template_bone_length_std = {
    "tail_base": 1.1,
    "spine_low": 1.8,
    "spine_mid": 1.0,
    "spine_high": 1.0,
    "left_ear": 1.55,
    "right_ear": 1.55,
    "forehead": 1.3,
    "nose_tip": 1.15,
    "left_fore_paw": 3.6,
    "right_fore_paw": 3.6,
    "left_hind_paw_back": 3.3,
    "left_hind_paw_front": 0.83,
    "right_hind_paw_back": 3.3,
    "right_hind_paw_front": 0.83,
    
}
# path to get back to spine_base from each keypoint
default_hierarchy = {
    # main body
    "spine_base": [],
    "spine_mid": ["spine_base"],
    "spine_high": ["spine_base"],
    "spine_low": ["spine_mid", "spine_base"],
    "tail_base": ["spine_low", "spine_mid", "spine_base"],
    # head
    "forehead": ["spine_high", "spine_base"],
    "nose_tip": ["forehead", "spine_high", "spine_base"],
    "left_ear": ["forehead", "spine_high", "spine_base"],
    "right_ear": ["forehead", "spine_high", "spine_base"],
    # fore limbs
    "left_fore_paw": ["spine_high", "spine_base"],
    "right_fore_paw": ["spine_high", "spine_base"],
    # hind paws
    "left_hind_paw_back": ["spine_low", "spine_mid", "spine_base"],
    "right_hind_paw_back": ["spine_low", "spine_mid", "spine_base"],
    "left_hind_paw_front": [
        "left_hind_paw_back",
        "spine_low",
        "spine_mid",
        "spine_base",
    ],
    "right_hind_paw_front": [
        "right_hind_paw_back",
        "spine_low",
        "spine_mid",
        "spine_base",
    ],
}


gimbal_skeleton = [
    ["tail_base", "spine_low"],
    ["spine_low", "spine_mid"],
    ["spine_mid", "spine_high"],
    ["spine_high", "left_ear"],
    ["spine_high", "right_ear"],
    ["spine_high", "forehead"],
    ["forehead", "nose_tip"],
    ["left_hind_paw_back", "left_hind_paw_front"],
    ["spine_low", "left_hind_paw_back"],
    ["right_hind_paw_back", "right_hind_paw_front"],
    ["spine_low", "right_hind_paw_back"],
    ["spine_high", "left_fore_paw"],
    ["spine_high", "right_fore_paw"],
]
