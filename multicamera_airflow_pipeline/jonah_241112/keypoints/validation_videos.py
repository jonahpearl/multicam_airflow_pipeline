import logging
import os
from pathlib import Path
import subprocess
import sys

import cv2
import h5py
import matplotlib.pyplot as plt
import multicam_calibration as mcc
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from tqdm.auto import tqdm

# load skeleton
from multicamera_airflow_pipeline.jonah_241112.skeletons.defaults import (
    dataset_info,
    kpt_dict,  # keypoint_name: id
    conf_thresholds_by_camera,
)
from multicamera_airflow_pipeline.utils.naming_utils import split_multicam_filename

logging.info("Python interpreter binary location:", sys.executable)
logger = logging.getLogger(__name__)

"""
TODO:
- change keypoints to x vs o for lo/hi confidence

"""

class KeypointVideoCreator:
    def __init__(
        self,
        predictions_2d_directory,
        predictions_triang_directory,
        predictions_gimbal_directory,
        camera_calibration_directory,
        raw_video_directory,
        output_directory_keypoint_vids,
        max_frames=2400,
        bbox_crop_size=(450, 450),
        recompute_completed=False,
        patterns_to_exclude_from_vids=["azure", "TRIM"],
    ):
        """
        Class to take the output of the keypoint pipeline and create videos with keypoints overlaid.

        Parameters:
        -----------
        predictions_2d_directory : str
            Path to the directory containing the 2D keypoint predictions.

        predictions_triang_directory : str
            Path to the directory containing the triangulated 3D keypoint predictions.

        predictions_gimbal_directory: str
            Path to the directory containing the gimbal inference predictions.

        camera_calibration_directory : str
            Path to the directory containing the camera calibration files.

        raw_video_directory: str
            Path to the directory containing the raw video files.

        output_directory_keypoint_vids : str
            Path to the directory where the output keypoint videos will be saved.

        max_frames : int
            Length of the output video in frames.

        recompute_completed : bool
            If True, the keypoint videos will be recreated even if they already exist in the output directory.
        """
        self.predictions_2d_directory = Path(predictions_2d_directory)
        self.predictions_triang_directory = Path(predictions_triang_directory)
        self.predictions_gimbal_directory = Path(predictions_gimbal_directory)
        self.camera_calibration_directory = Path(camera_calibration_directory)
        self.raw_video_directory = Path(raw_video_directory)
        self.output_directory_keypoint_vids = Path(output_directory_keypoint_vids)
        self.max_frames = max_frames
        self.recompute_completed = recompute_completed
        self.patterns_to_exclude_from_vids = patterns_to_exclude_from_vids
        self.bbox_crop_size = bbox_crop_size

        # Initialize keypoint and skeleton information from dataset_info
        self.keypoint_info = dataset_info["keypoint_info"]
        self.skeleton_info = dataset_info["skeleton_info"]
        self.keypoint_idx = {
            value["name"]: key for key, value in self.keypoint_info.items()
        }

        # Order of keypoints as predicted
        self.keypoints = np.array(
            [self.keypoint_info[i]["name"] for i in self.keypoint_info.keys()]
        )
        self.n_keypoints = len(self.keypoints)

    def check_if_validation_vids_exist(self):
        # checks a log saved in the output directory to see if the validation videos have already been created

        if (
            self.output_directory_keypoint_vids / "keypoint_videos_completed.log"
        ).exists():
            return True
        else:
            return False

    def set_completed(self):
        with open(
            self.output_directory_keypoint_vids / "keypoint_videos_completed.log", "w"
        ) as f:
            f.write("Keypoint vids completed")

    def load_video_filenames(self):
        # grab all the video files
        self.video_files = {}
        for camera in self.cameras:
            self.video_files[camera] = list(
                self.raw_video_directory.glob(f"*{camera}*.mp4")
            )
            self.video_files[camera] = [v for v in self.video_files[camera] if not any([p in v.name for p in self.patterns_to_exclude_from_vids])]
            if len(self.video_files[camera]) == 0:
                raise ValueError(f"No video files found for camera {camera}")
            elif len(self.video_files[camera]) > 1:
                raise ValueError(f"Multiple video files found for camera {camera}")
            else:
                self.video_files[camera] = self.video_files[camera][0]

    def load_2D_prediction_filenames(self):
        # grab all the predictions 2D h5 files
        predictions_2d_files = list(self.predictions_2d_directory.glob("*.h5"))
        filename_splits = [split_multicam_filename(i) for i in predictions_2d_files]
        cam = [i["camera"] for i in filename_splits]
        frame = [i["start_frame"] for i in filename_splits]
        # cam = [i.stem.split(".")[1] for i in predictions_2d_files]
        # frame = [int(i.stem.split(".")[2]) for i in predictions_2d_files]
        self.recording_predictions = pd.DataFrame(
            {"camera": cam, "frame": frame, "file": predictions_2d_files}
        )
        # Filter out cameras not present in data
        self.recording_predictions = self.recording_predictions[
            self.recording_predictions.camera.isin(self.cameras)
        ]

    def load_triang_prediction_filenames(self):
        predictions_triang_files = list(
            self.predictions_triang_directory.glob("predictions_3d*.mmap")
        )
        confidences_triang_files = list(
            self.predictions_triang_directory.glob("confidences_3d*.mmap")
        )
        assert len(predictions_triang_files) == len(confidences_triang_files) == 1
        self.triang_predictions_file = predictions_triang_files[0]
        self.triang_confidences_file = confidences_triang_files[0]

    def load_gimbal_inference_filenames(self):
        predictions_gimbal_files = list(
            self.predictions_gimbal_directory.glob("gimbal.float*.mmap")
        )
        assert len(predictions_gimbal_files) == 1
        self.gimbal_predictions_file = predictions_gimbal_files[0]
        keypoints_used_in_gimbal_file = list(
            self.predictions_gimbal_directory.glob("keypoints_order_gimbal.npy")
        )
        assert len(keypoints_used_in_gimbal_file) == 1
        keypoints_in_gimbal = list(set(np.load(keypoints_used_in_gimbal_file[0])))
        self.keypoint_idxs_missing_in_gimbal = []
        for keypoint in kpt_dict:
            if keypoint not in keypoints_in_gimbal:
                self.keypoint_idxs_missing_in_gimbal.append(kpt_dict[keypoint])

    def load_2D_predictions(self):
        # Load up to max_frames of 2D predictions
        self.predictions_2d = {}
        for camera in self.cameras:
            keypoint_coords = []
            keypoint_conf = []
            detection_coords = []
            frames_loaded = 0
            while frames_loaded < self.max_frames:
                # Get the next file
                next_file = (
                    self.recording_predictions[
                        (self.recording_predictions.camera == camera)
                        & (self.recording_predictions.frame >= frames_loaded)
                    ]
                    .sort_values("frame")
                    .iloc[0]
                )
                # Load the predictions
                with h5py.File(next_file.file, "r") as file:
                    _keypoint_coords = np.array(
                        file["keypoint_coords"]
                    )  # shape: (n_frames, n_keypoints, 2)
                    _keypoint_conf = np.array(file["keypoint_conf"])
                    _detection_coords = np.array(file["detection_coords"])
                _keypoint_conf[_keypoint_conf > 1] = 1
                keypoint_conf.append(_keypoint_conf)
                keypoint_coords.append(_keypoint_coords)
                detection_coords.append(_detection_coords)
                frames_loaded += len(_keypoint_coords)

            self.predictions_2d[camera] = {
                "keypoint_coords": np.concatenate(keypoint_coords, axis=0).squeeze(),
                "keypoint_conf": np.concatenate(keypoint_conf, axis=0).squeeze(),
                "detection_coords": np.concatenate(detection_coords, axis=0).squeeze(),
            }

    def load_triang_reproj_predictions(self):
        # Load max_frames of 2D reprojections
        self.predictions_triang = {}
        keypoint_coords = load_memmap_from_filename(self.triang_predictions_file)  # shape: (n_frames, n_keypoints, 3)
        keypoint_conf = load_memmap_from_filename(self.triang_confidences_file)
        for iCamera, camera in enumerate(self.cameras):
            these_coords_3D = keypoint_coords[:self.max_frames, :, :]
            these_confs = keypoint_conf[:self.max_frames, :]

            # Reproject the coords into 2D
            extrinsics = self.all_extrinsics[iCamera]
            camera_matrix, dist_coefs = self.all_intrinsics[iCamera]
            these_coords_2D = mcc.project_points(
                these_coords_3D,
                extrinsics=extrinsics,
                camera_matrix=camera_matrix,
                dist_coefs=dist_coefs,
            )
            self.predictions_triang[camera] = {
                "keypoint_coords": these_coords_2D,
                "keypoint_conf": these_confs,
            }

    def load_gimbal_reproj_predictions(self):
        # Load max_frames of 2D reprojections from gimbal output
        self.predictions_gimbal = {}
        keypoint_coords = load_memmap_from_filename(self.gimbal_predictions_file)  # shape: (n_frames, n_keypoints, 3)
        for iCamera, camera in enumerate(self.cameras):
            these_coords_3D = keypoint_coords[:self.max_frames, :, :]

            # Reproject the coords into 2D
            extrinsics = self.all_extrinsics[iCamera]
            camera_matrix, dist_coefs = self.all_intrinsics[iCamera]
            these_coords_2D = mcc.project_points(
                these_coords_3D,
                extrinsics=extrinsics,
                camera_matrix=camera_matrix,
                dist_coefs=dist_coefs,
            )

            # Median smooth with n=5 (probably would also smooth this way in any analysis using the gimbal points)
            these_coords_2D = median_filter(these_coords_2D, size=(5, 1, 1))

            # Add nans to any keypoints not in gimbal
            # by expanding these_coords_2D on the keypoint axis with nans.
            for idx in self.keypoint_idxs_missing_in_gimbal:
                these_coords_2D = np.insert(these_coords_2D, idx, np.nan, axis=1)
            
            # Store data
            self.predictions_gimbal[camera] = {
                "keypoint_coords": these_coords_2D,
            }

    def get_gimbal_centroids(self):
        detection_coords_by_camera = {}
        for camera in self.cameras:
            reproj_coords = self.predictions_gimbal[camera]["keypoint_coords"]
            centroids = np.nanmean(reproj_coords, axis=1)
            centroids = nan_to_preceding(centroids)
            detection_coords_by_camera[camera] = centroids
        self.detection_coords_by_camera = detection_coords_by_camera

    def create_2D_keypoint_conf_plots(self):
        all_kp_confs = np.stack(
            [self.predictions_2d[cam]["keypoint_conf"] for cam in self.cameras], axis=-1
        )
        max_kp_confs_per_frame = np.max(all_kp_confs, axis=-1)
        plt.figure()
        plt.matshow(
            max_kp_confs_per_frame.T, aspect="auto", cmap="PiYG", vmin=0, vmax=1
        )
        cbar = plt.colorbar()
        cbar.set_label("Max keypoint confidence")
        cbar.set_ticks([0, 0.5, 1])
        plt.ylabel("Keypoint")
        plt.xlabel("Frame")
        plt.title(
            f"Max keypoint confidences across cameras\n{self.recording_predictions.iloc[0].file.stem}"
        )
        plt.savefig(self.output_directory_keypoint_vids / "max_keypoint_confs.png")
        plt.close()

    def create_2D_keypoint_videos(self):
        for camera in self.cameras:
            keypoint_coords = self.predictions_2d[camera]["keypoint_coords"]
            keypoint_conf = self.predictions_2d[camera]["keypoint_conf"]
            bbox_coords = self.predictions_2d[camera]["detection_coords"]
            conf_thresholds = conf_thresholds_by_camera["side"] if "side" in camera else conf_thresholds_by_camera[camera]
            generate_keypoint_video(
                output_directory=self.output_directory_keypoint_vids,
                video_path=self.video_files[camera],
                keypoint_coords=keypoint_coords,
                keypoint_conf=keypoint_conf,
                keypoint_info=dataset_info["keypoint_info"],
                vid_suffix="with_2D_keypoints",
                detection_coords=bbox_coords,
                skeleton_info=dataset_info["skeleton_info"],
                conf_thresholds=conf_thresholds,
                max_frames=self.max_frames,
            )

    def create_reproj_triang_keypoint_videos(self):
        for camera in self.cameras:
            keypoint_coords = self.predictions_triang[camera]["keypoint_coords"]
            keypoint_conf = self.predictions_triang[camera]["keypoint_conf"]

            generate_keypoint_video(
                output_directory=self.output_directory_keypoint_vids,
                video_path=self.video_files[camera],
                keypoint_coords=keypoint_coords,
                keypoint_conf=keypoint_conf,
                keypoint_info=dataset_info["keypoint_info"],
                vid_suffix="with_triang_keypoints",
                skeleton_info=dataset_info["skeleton_info"],
                conf_thresholds=0.5,
                max_frames=self.max_frames,
            )

    def create_reproj_gimbal_keypoint_videos(self):
        for camera in self.cameras:
            keypoint_coords = self.predictions_gimbal[camera]["keypoint_coords"]

            generate_keypoint_video(
                output_directory=self.output_directory_keypoint_vids,
                video_path=self.video_files[camera],
                keypoint_coords=keypoint_coords,
                keypoint_conf=None,
                keypoint_info=dataset_info["keypoint_info"],
                vid_suffix="with_gimbal_keypoints",
                skeleton_info=dataset_info["skeleton_info"],
                max_frames=self.max_frames,
            )

    def crop_and_stitch_2D_keypoint_videos(self):
        crop_and_stich_vids(
            output_directory=self.output_directory_keypoint_vids,
            single_vid_suffix="with_2D_keypoints",  # Suffix to identify the single videos to be stitched together
            bbox_crop_size=self.bbox_crop_size,
            detection_coords_by_camera=self.detection_coords_by_camera,
            max_frames=self.max_frames,
        )

    def crop_and_stitch_reproj_triang_keypoint_videos(self):
        # Use the centroid of the triang'd kps as the center of the bbox
        # detection_coords_by_camera = {}
        # for camera in self.cameras:
        #     reproj_coords = self.predictions_triang[camera]["keypoint_coords"]
        #     centroids = np.nanmean(reproj_coords, axis=1)
        #     centroids = nan_to_preceding(centroids)
        #     detection_coords_by_camera[camera] = centroids

        crop_and_stich_vids(
            output_directory=self.output_directory_keypoint_vids,
            single_vid_suffix="with_triang_keypoints",  # Suffix to identify the single videos to be stitched together
            detection_coords_by_camera=self.detection_coords_by_camera,
            bbox_crop_size=self.bbox_crop_size,
            max_frames=self.max_frames,
        )

    def crop_and_stitch_gimbal_keypoint_videos(self):
        # Use the centroid of the coords as the center of the bbox
        crop_and_stich_vids(
            output_directory=self.output_directory_keypoint_vids,
            single_vid_suffix="with_gimbal_keypoints",  # Suffix to identify the single videos to be stitched together
            detection_coords_by_camera=self.detection_coords_by_camera,
            bbox_crop_size=self.bbox_crop_size,
            max_frames=self.max_frames
        )

    def run(self):
        # Check if already completed
        if not self.recompute_completed:
            if self.check_if_validation_vids_exist():
                logger.info("Triangulation already exists")
                return

        # Create output dir if needed
        if not self.output_directory_keypoint_vids.exists():
            self.output_directory_keypoint_vids.mkdir(parents=True)

        # Load calibration + camera info
        self.all_extrinsics, self.all_intrinsics, camera_names = mcc.load_calibration(
            self.camera_calibration_directory,
            load_format="jarvis",
        )
        self.cameras = camera_names
        self.n_cameras = len(self.cameras)
        logging.info(f"\t n_cameras {self.n_cameras}")

        # Load the video filenames
        self.load_video_filenames()

        # Load the 2D predictions
        self.load_2D_prediction_filenames()
        self.load_2D_predictions()

        # Load the triangulated 3D predictions
        self.load_triang_prediction_filenames()
        self.load_triang_reproj_predictions()
        
        # Load gimbal inference predictions
        self.load_gimbal_inference_filenames()
        self.load_gimbal_reproj_predictions()

        # Get centroids from GIMBAL to use across all videos
        self.get_gimbal_centroids()

        # Create the 2D keypoint videos
        logging.info("Making 2D keypoint videos")
        self.create_2D_keypoint_conf_plots()
        self.create_2D_keypoint_videos()
        logging.info("Stitching 2D keypoint videos")
        self.crop_and_stitch_2D_keypoint_videos()
        

        # Create the triangulated keypoint videos
        logging.info("Making 3D triang videos")
        self.create_reproj_triang_keypoint_videos()
        logging.info("Stitching 3D triang videos")
        self.crop_and_stitch_reproj_triang_keypoint_videos()

        # Create the gimbal keypoint videos
        logging.info("Making gimbal videos")
        self.create_reproj_gimbal_keypoint_videos()
        self.crop_and_stitch_gimbal_keypoint_videos()

        logging.info("Stitching rows together...")
        crop_and_stitch_rows_of_vids(
            output_directory=self.output_directory_keypoint_vids,
            bbox_crop_size=self.bbox_crop_size,
        )

        # Compress all the videos
        # (Runs at ~10 fps --> adds another 7200 frames / 10 fps = 720s = 12 minutes x 6 vids = ~1 hr)
        logging.info("Compressing videos")
        for vid in self.output_directory_keypoint_vids.glob("*.mp4"):
            compressed_vid = vid.with_name(vid.stem + "_compressed.mp4")
            compress_vid_via_ffmpeg(
                vid,
                compressed_vid,
                crf=23,
                preset="fast",  # runs at ~10 fps (which is kinda slow) but compresses ~10x which is nice for speeding up subsequent downloads of the QC vids.
                recompute_completed=self.recompute_completed,
            )
        
            # Remove the original
            os.remove(vid)

            # Rename the compressed file
            compressed_vid.rename(vid)
        
        # Mark as completed
        logging.info("Marking as completed")
        self.set_completed()

        return


def compress_vid_via_ffmpeg(input_vid, output_vid, crf=23, preset="fast", recompute_completed=False):
    """
    Compresses a video file using ffmpeg.

    Parameters:
    -----------
    input_vid : Path
        Path to the input video file.

    output_vid : Path
        Path to the output video file.

    crf : int
        Constant Rate Factor (CRF) value for the video compression. Lower values result in higher quality videos.

    preset : str
        Preset for the video compression. Options are: 'ultrafast', 'superfast', 'veryfast', 'faster', 'fast', 'medium',
    """
    
    # Check if the output file already exists
    if output_vid.exists() and not recompute_completed:
        logging.info(f"Output file already exists: {output_vid}")
        return

    # Run the ffmpeg command for video compression
    command = [
        "/n/app/ffmpeg/3.3.3/ffmpeg",
        "-y",
        "-i",
        str(input_vid),
        "-c:v",
        "libx264",
        "-crf",
        str(crf),
        "-preset",
        preset,
        str(output_vid),
    ]
    logging.info(f"Compressing video: {input_vid}")
    subprocess.Popen(command).wait()
    return


def generate_keypoint_video(
    output_directory: Path,
    video_path: Path,
    keypoint_coords: np.ndarray,
    keypoint_conf: np.ndarray,  # New parameter for keypoint confidence
    keypoint_info: dict,
    skeleton_info: dict,
    vid_suffix: str,
    detection_coords: np.ndarray = None,
    conf_thresholds: dict = None,
    max_frames=None,
    overwrite=False,
):
    """
    Generates a video with keypoint predictions overlaid on the original video frames.

    Parameters:
    -----------
    output_directory : Path
        Directory where the output video will be saved.

    video_path : Path
        Path to the input video file.

    keypoint_coords : np.ndarray
        Array of shape (#frames, #keypoints, 2) containing the coordinates of keypoints for each frame.

    keypoint_conf : np.ndarray
        Array of shape (#frames, #keypoints) containing the confidence values (0-1) for each keypoint in each frame.

    keypoint_info : dict
        Dictionary containing information about the keypoints. Each key in the dictionary represents a keypoint ID, and the
        value is another dictionary with the following structure:
        {
            'name': str,       # Keypoint name
            'id': int,         # Keypoint ID
            'color': list,     # RGB color for the keypoint [R, G, B]
            'type': str,       # Keypoint type (e.g., 'upper', 'lower')
            'swap': str        # Name of the corresponding left/right keypoint to be swapped (for symmetry)
        }

    skeleton_info : dict
        Dictionary containing information about the skeleton. Each key in the dictionary represents a skeleton link ID, and
        the value is another dictionary with the following structure:
        {
            'link': tuple,     # Tuple containing the names of the two keypoints that form the link
            'id': int,         # Link ID
            'color': list      # RGB color for the link [R, G, B]
        }

    vid_suffix : str
        Suffix to add to the video file name. Ie "with_2D_keypoints" or "with_3D_keypoints"

    detection_coords : np.ndarray
        Array of shape (#frames, 4) containing the bounding box coordinates (x1, y1, x2, y2) for each frame. If provided,
        the bounding box will be drawn on the video frames.

    conf_thresholds : dict | int | None
        Dictionary containing the confidence thresholds (for 2D prediction) for each keypoint type.
        If a keypoint's confidence is below the threshold, it will be drawn as a small "x".
        Possible keypoint types:
            "tail", "spine", "hind_paw", "fore_paw", "ear", "forehead",  "nose_tip", 
        If an integer is provided, it will be used as the threshold for all keypoints.
        If None, all keypoints will be drawn with a circle.

    max_frames : int
        Maximum number of frames to process. If None, all frames will be processed.

    overwrite : bool
        If True, the output video will be overwritten if it already exists.

    Returns:
    --------
    None
        The function saves the output video with keypoints and skeletons overlaid to the specified output directory.

    Raises:
    -------
    ValueError
        If the input video cannot be opened.

    Example:
    --------
    output_directory = Path('/output/directory')
    video_path = Path('/path/to/video.mp4')
    keypoint_coords = np.load('keypoint_coords.npy')  # Load your keypoints array
    keypoint_conf = np.load('keypoint_conf.npy')  # Load your keypoint confidence array
    keypoint_info = {
        0: {'name': 'nose_tip', 'id': 0, 'color': [120, 184, 181], 'type': 'upper', 'swap': ''},
        # Add other keypoints as needed
    }
    skeleton_info = {
        0: {'link': ('tail_base', 'spine_low'), 'id': 0, 'color': [173, 160, 183]},
        # Add other links as needed
    }

    generate_keypoint_video(output_directory, video_path, keypoint_coords, keypoint_conf, keypoint_info, skeleton_info)
    """

    print(f"Generating video {vid_suffix} from {os.path.basename(video_path)}")
    filename_parts = split_multicam_filename(video_path)
    camera = filename_parts["camera"]

    # Check if the output video already exists
    output_path = output_directory / (video_path.stem + "." + vid_suffix + ".mp4")
    if output_path.exists() and not overwrite:
        logging.info(f"Output video already exists: {output_path}")
        return

    # Open the input video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create the VideoWriter object
    output_path = output_directory / (video_path.stem + "." + vid_suffix + ".mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height))

    frame_idx = 0
    if total_frames < 0 and max_frames is None:
        raise ValueError(
            "Could not determine total number of frames in the video -- please specify max_frames."
        )
    elif total_frames < 0:
        total_frames = max_frames
    elif max_frames is not None:
        total_frames = np.min([max_frames, total_frames])

    logging.info(f"Total frames: {total_frames}")

    with tqdm(total=total_frames, desc="Processing frames") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Create an overlay for drawing
            # overlay = frame.copy()

            # Draw keypoints
            for kp_idx, kp_info in keypoint_info.items():
                if (
                    frame_idx < len(keypoint_coords)
                    and kp_idx < keypoint_coords.shape[1]
                ):

                    # If using confidence thresholds per keypoint, find the threshold
                    if conf_thresholds is not None and isinstance(conf_thresholds, dict):
                        this_conf_thresh = [val for keypoint_type,val in conf_thresholds.items() if keypoint_type in kp_info["name"]]
                    elif isinstance(conf_thresholds, int):
                        this_conf_thresh = conf_thresholds
                    else:
                        this_conf_thresh = 0

                    # Get the coords. If nans, skip.
                    x, y = keypoint_coords[frame_idx, kp_idx]
                    if np.isnan(x) or np.isnan(y):
                        continue

                    # If no confidence values provided, just set to 1.
                    if keypoint_conf is not None:
                        conf = keypoint_conf[frame_idx, kp_idx]
                    else:
                        conf = 1

                    # Draw the keypoint  
                    color = tuple(kp_info["color"])
                    if conf > this_conf_thresh:  # Only draw if confidence is greater than 0
                        frame = cv2.circle(
                            frame,
                            (int(x), int(y)),
                            radius=5,
                            color=color,
                            thickness=-1,
                        )
                    else:
                        frame = cv2.drawMarker(
                            frame,
                            (int(x), int(y)),
                            color=color,
                            markerType=cv2.MARKER_TILTED_CROSS,
                            markerSize=12,
                            thickness=2,
                        )

            # Draw skeleton
            for link_info in skeleton_info.values():
                kp1_name, kp2_name = link_info["link"]
                kp1_id = next(
                    (
                        kp["id"]
                        for kp in keypoint_info.values()
                        if kp["name"] == kp1_name
                    ),
                    None,
                )
                kp2_id = next(
                    (
                        kp["id"]
                        for kp in keypoint_info.values()
                        if kp["name"] == kp2_name
                    ),
                    None,
                )

                if kp1_id is not None and kp2_id is not None:
                    if (
                        frame_idx < len(keypoint_coords)
                        and kp1_id < keypoint_coords.shape[1]
                        and kp2_id < keypoint_coords.shape[1]
                    ):
                        x1, y1 = keypoint_coords[frame_idx, kp1_id]
                        x2, y2 = keypoint_coords[frame_idx, kp2_id]
                        color = tuple(link_info["color"])
                        if np.isnan(x1) or np.isnan(y1) or np.isnan(x2) or np.isnan(y2):
                            continue
                        frame = cv2.line(
                            frame,
                            (int(x1), int(y1)),
                            (int(x2), int(y2)),
                            color=color,
                            thickness=2,
                        )

            # Find centroid of bounding box
            # x1, y1, x2, y2 = detection_coords[frame_idx, 0, :]
            # centroid = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            # frame = cv2.circle(
            #     frame, centroid, radius=4, color=(0, 255, 0), thickness=-1
            # )

            # Draw the detection bounding box on the frame
            # if frame_idx < len(detection_coords):
            #     x1, y1, x2, y2 = detection_coords[frame_idx,0,:]
            #     frame = cv2.rectangle(
            #         frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2
            #     )

            # Write the frame number in the top left corner
            frame = cv2.putText(
                frame,
                str(frame_idx),
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # Write the camera name below the frame number
            frame = cv2.putText(
                frame,
                camera,
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # Write the frame with keypoints and skeletons to the output video
            out.write(frame)
            frame_idx += 1
            pbar.update(1)
            if max_frames and frame_idx >= max_frames:
                break

    # Release video objects
    cap.release()
    out.release()
    logging.info(f"Video saved to: {output_path}")
    return


def crop_and_stich_vids(
    output_directory,
    single_vid_suffix,
    bbox_coords_by_camera=None,
    detection_coords_by_camera=None,
    bbox_crop_size=(400, 400),
    max_frames=None,
    overwrite=False,
):
    """
    Take keypoint videos and crop the mouse out, and stitch together the cropped videos into one row.

    Parameters:
    -----------
    output_directory : Path
        Directory where the single videos will be found + output video will be saved.

    single_vid_suffix : str
        Suffix to identify the single videos to be stitched together.

    bbox_coords_by_camera : dict[np.ndarray] or None
        Dictionary containing the bounding box coordinates for each camera. The keys are camera names and the values are
        numpy arrays of shape (#frames, 4) containing the bounding box coordinates (x1, y1, x2, y2) for each frame.
        If None, must provide detection coordinates instead, which wil be treated as centroids.

    detection_coords_by_camera : dict[np.ndarray] or None
        Dictionary containing the detection coordinates for each camera. The keys are camera names and the values are
        numpy arrays of shape (#frames, 4) containing the detection (ie centroid) coordinates (x, y) for each frame.
        If None, must provide bbox coordinates instead, which will be used to infer a centroid + crop
        (the bboxes from mmpose aren't uniform size, so we infer centroid + crop to standard size).

    """
    print(f"Cropping and stitching videos: {single_vid_suffix}")

    assert (
        bbox_coords_by_camera is not None or detection_coords_by_camera is not None
    ), "Must provide either bbox or detection coordinates."
    assert (
        bbox_coords_by_camera is None or detection_coords_by_camera is None
    ), "Must provide either bbox or detection coordinates, not both."

    out_vids = list(output_directory.glob(f"*{single_vid_suffix}.mp4"))
    # timestamp, cam, vid_suffix = out_vids[0].stem.split(".")
    filename_info = split_multicam_filename(out_vids[0], mode="validation_videos")
    stitched_vid_name = ".".join([filename_info["rec_name"], "stitched", filename_info["suffix"], "mp4"])

    # Check if the output video already exists
    out_vid_path = output_directory / Path(stitched_vid_name)
    if out_vid_path.exists() and not overwrite:
        logging.info(f"Output video already exists: {out_vid_path}")
        return

    # Get the total number of frames to use
    tmp_cap = cv2.VideoCapture(str(out_vids[0]))
    total_frames = int(tmp_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 0 and max_frames is None:
        raise ValueError(
            "Could not determine total number of frames in the video -- please specify max_frames."
        )
    elif total_frames < 0:
        total_frames = max_frames
    elif max_frames is not None:
        total_frames = np.min([max_frames, total_frames])
    tmp_cap.release()

    # Calculate bbox centroids for the cropping
    bbox_centroids_by_camera = {}
    if bbox_coords_by_camera is not None:
        for vid in out_vids:
            # recording_id, camera, frame, ext = os.path.basename(vid).split(".")
            camera = split_multicam_filename(vid, mode="validation_videos")["camera"]
            detn_coords = bbox_coords_by_camera[camera]
            bbox_centroids_by_camera[camera] = np.array(
                [[(x1 + x2) / 2, (y1 + y2) / 2] for x1, y1, x2, y2 in detn_coords]
            )
            # Apply median filter smoothing to reduce jitter
            bbox_centroids_by_camera[camera] = median_filter(
                bbox_centroids_by_camera[camera], size=(12, 1)
            )
    elif detection_coords_by_camera is not None:
        for vid in out_vids:
            # recording_id, camera, frame, ext = os.path.basename(vid).split(".")
            camera = split_multicam_filename(vid, mode="validation_videos")["camera"]
            bbox_centroids_by_camera[camera] = median_filter(
                detection_coords_by_camera[camera], size=(12, 1)
            )

    # Open the output video
    out_vid_path = output_directory / Path(stitched_vid_name)
    logging.info(f"Output video path: {out_vid_path}")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_frame_size = (bbox_crop_size[0] * len(out_vids), bbox_crop_size[1])
    out = cv2.VideoWriter(str(out_vid_path), fourcc, 30, output_frame_size)

    with tqdm(total=total_frames, desc="Processing frames") as pbar:
        # Open the input videos
        cap_by_camera = {}
        for vid in out_vids:
            cap = cv2.VideoCapture(str(vid))
            camera = split_multicam_filename(vid, mode="validation_videos")["camera"]
            cap_by_camera[camera] = cap

        frame_idx = 0
        while True:
            frames = []
            for camera, cap in cap_by_camera.items():
                # Read the frame
                ret, frame = cap.read()
                if not ret:
                    break

                # Crop the frame
                x, y = bbox_centroids_by_camera[camera][frame_idx]
                x1, y1 = x - bbox_crop_size[0] // 2, y - bbox_crop_size[1] // 2
                x2, y2 = x + bbox_crop_size[0] // 2, y + bbox_crop_size[1] // 2
                frame = frame[int(y1) : int(y2), int(x1) : int(x2)]

                # Write the camera name in the top left corner
                frame = cv2.putText(
                    frame,
                    camera,
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                if camera == "bottom":
                    # Also add the frame number below the camera name
                    frame = cv2.putText(
                        frame,
                        str(frame_idx),
                        (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                frames.append(frame)

            if not ret:
                break

            # Stitch the frames together
            stitched_frame = np.zeros(
                (bbox_crop_size[1], bbox_crop_size[0] * len(out_vids), 3),
                dtype=np.uint8,
            )
            for i, frame in enumerate(frames):
                stitched_frame[
                    0 : frame.shape[0], i * frame.shape[1] : (i + 1) * frame.shape[1]
                ] = frame

            # Write the stitched frame to the output video
            out.write(stitched_frame)

            # Loop control
            frame_idx += 1
            pbar.update(1)
            if max_frames and frame_idx >= max_frames:
                break


def crop_and_stitch_rows_of_vids(output_directory, bbox_crop_size=(400, 400), overwrite=False):
    stitched_row_vids = list(output_directory.glob("*stitched.*.mp4"))
    filename_info = split_multicam_filename(stitched_row_vids[0], mode="validation_videos")
    out_vid_name = ".".join([filename_info["rec_name"], "stitched_rows", "mp4"])
    out_vid = output_directory / Path(out_vid_name)

    # Check if already done
    if out_vid.exists() and not overwrite:
        logging.info(f"Output video already exists: {out_vid}")
        return

    # Prepare video for writing
    n_cams = len(list(output_directory.glob("*with_2D_keypoints*.mp4"))) - 1  # Subtract 1 for the stitched video
    output_frame_size = (bbox_crop_size[0] * n_cams, bbox_crop_size[1] * len(stitched_row_vids))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(out_vid), fourcc, 30, output_frame_size)

    # Open the input videos in order of 2D keypoints, triangulation, gimbal
    preds_2d_vid = [v for v in stitched_row_vids if "with_2D_keypoints" in v.stem]
    preds_triang_vid = [v for v in stitched_row_vids if "with_triang_keypoints" in v.stem]
    preds_gimbal_vid = [v for v in stitched_row_vids if "with_gimbal_keypoints" in v.stem]
    vid_caps = []
    for vid in preds_2d_vid + preds_triang_vid + preds_gimbal_vid:
        cap = cv2.VideoCapture(str(vid))
        vid_caps.append(cap)
    
    total_frames = int(vid_caps[0].get(cv2.CAP_PROP_FRAME_COUNT))
    with tqdm(total=total_frames, desc="Processing frames") as pbar:
        frame_idx = 0
        while True:
            rows = []
            for cap in vid_caps:
                ret, row = cap.read()
                if not ret:
                    break
                rows.append(row)
            if not ret:
                break

            # Stitch the frames together
            stitched_frame = np.zeros(
                (bbox_crop_size[1] * len(stitched_row_vids), bbox_crop_size[0] * n_cams, 3),
                dtype=np.uint8,
            )
            
            for i, row in enumerate(rows):
                # import pdb; pdb.set_trace()
                # Add each row to the stitched frame
                stitched_frame[
                    i * bbox_crop_size[1] : (i + 1) * bbox_crop_size[1], :, :
                ] = row

            # Write the stitched frame to the output video
            out.write(stitched_frame)

            # Loop control
            frame_idx += 1
            pbar.update(1)
    
    # Release video objects
    for cap in vid_caps:
        cap.release()
    
    # Release the output video
    out.release()

    return

def load_memmap_from_filename(filename):
    # Extract the metadata from the filename
    parts = filename.name.rsplit(".", 4)  # Split the filename into parts
    dtype_str = parts[-3]  # Get the dtype part of the filename
    shape_str = parts[-2]  # Get the shape part of the filename
    shape = tuple(
        map(int, shape_str.split("x"))
    )  # Convert shape string to a tuple of integers
    # Load the array using numpy memmap
    array = np.memmap(filename, dtype=dtype_str, mode="r", shape=shape)
    return array


def nan_to_preceding(arr):
    # Make a copy of the array to avoid modifying the original array
    result = arr.copy()

    # Iterate over each element in the first dimension
    for i in range(1, arr.shape[0]):
        mask = np.isnan(result[i])  # Identify the NaN values
        result[i][mask] = result[i - 1][mask]  # Replace NaNs with preceding values

    return result
