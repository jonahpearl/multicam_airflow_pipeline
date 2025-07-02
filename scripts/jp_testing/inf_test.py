
# load params
import yaml
params_file = "/n/groups/datta/kpts_pipeline/jonah_241112/jobs/2D_predictions/20250613_J10403_20250616_151456_089403/20250613_J10403_2d_predictions_25-06-16-2025-14-56-097063.params.yaml"
config_file = "/n/groups/datta/Jonah/Local_code_groups/6cam_repos/multicam_airflow_pipeline/multicamera_airflow_pipeline/jonah_241112/default_config_v2.yaml"

params = yaml.safe_load(open(params_file, 'r'))
config = yaml.safe_load(open(config_file, 'r'))

# covert models to tensorrt
# from multicamera_airflow_pipeline.jonah_241112.keypoints.tensorrt import RTMModelConverter
# model_converter = RTMModelConverter(
#     tensorrt_output_directory = params["tensorrt_model_directory"],
#     **config["tensorrt_conversion"]
# )
# model_converter.run()

# grab sync cameras function
from multicamera_airflow_pipeline.jonah_241112.keypoints.predict_2D import Inferencer2D
inferencer = Inferencer2D(
    # **params,
    expected_video_length_frames=216008,
    output_directory_predictions="/n/groups/datta/Jonah/20250513_PAG_stim/raw_data/J10403/20250613_J10403/trimmed_vids/results",
    recording_directory="/n/groups/datta/Jonah/20250513_PAG_stim/raw_data/J10403/20250613_J10403/trimmed_vids",
    # **config["prediction_2d"],
    pose_estimator_config="/n/groups/datta/6cam_keypoint_networks/mm_pose/Jonah/20241216_v2/rtmpose/rtmpose-m_8xb64-210e_ap10k-256x256_24-12-16-14-52-59/config.py",
    pose_estimator_checkpoint="/n/groups/datta/6cam_keypoint_networks/mm_pose/Jonah/20241216_v2/rtmpose/rtmpose-m_8xb64-210e_ap10k-256x256_24-12-16-14-52-59/best_PCK_epoch_90.pth",
    detector_config="/n/groups/datta/6cam_keypoint_networks/mm_pose/Jonah/20241030_v1/rtmdet/rtmdet_small_8xb32-300e_coco_chronic_24-10-31-12-48-26/config.py",
    detector_checkpoint="/n/groups/datta/6cam_keypoint_networks/mm_pose/Jonah/20241030_v1/rtmdet/rtmdet_small_8xb32-300e_coco_chronic_24-10-31-12-48-26/epoch_63.pth",
    n_keypoints=15,
    n_animals=1,
    detection_interval=1,
    use_motpy=True,
    n_motpy_tracks=3,
    ignore_log_files=True,
    # use_tensorrt=False,
    use_tensorrt=True,
    tensorrt_model_directory="/n/groups/datta/kpts_pipeline/jonah_241112/results/tensorrt",
    tensorrt_rtmpose_model_name='rtmpose-m_8xb64-210e_ap10k-256x256_24-12-16-14-52-59',
    tensorrt_rtmdetection_model_name='rtmdet_small_8xb32-300e_coco_chronic_24-10-31-12-48-26',
    patterns_to_exclude_from_vids=[],
)
inferencer.run()
