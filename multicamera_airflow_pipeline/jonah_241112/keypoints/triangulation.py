# import spikeinterface as si
import logging
import os
from pathlib import Path
import shutil
import sys
import tempfile

import h5py
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import multicam_calibration as mcc
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter
from tqdm.auto import tqdm

from multicamera_airflow_pipeline.jonah_241112.skeletons.defaults import (
    dataset_info,
    parents_dict,
    conf_thresholds_by_camera,
    keypoint_info,
)
from multicamera_airflow_pipeline.utils.naming_utils import split_multicam_filename


print("Python interpreter binary location:", sys.executable)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Triangulator:
    def __init__(
        self,
        predictions_2d_directory,  # Directory where 2D predictions are stored for this video
        output_directory_triangulation,  # Directory to save resulting memmap files
        camera_sync_file,  # CSV file with camera sync data
        expected_frames_per_video,  # Expected number of frames per video
        camera_calibration_directory,  # Directory with calibration data in jarvis format
        n_jobs=5,  # Number of parallel jobs
        keep_top_k=3,  # Number of top keypoints to keep
        leave_one_out_center_threshold_mm=50,  # Threshold distance for leave-one-out center in mm
        threshold_distance_from_parent_mm=50,  # Threshold distance from parent keypoint in mm
        mmap_dtype="float32",  # Data type for memory-mapped files
        print_nans=True,  # Flag to print NaNs
        mean_filt_samples=11,  # Number of samples for mean filtering (~100ms at 120fps)
        mean_filt_distance_thresh_px=150,  # Distance threshold for mean filtering in pixels
        perform_top_k_filtering=False,
        recompute_completed=False,
        suppress_assertion=False,
        sub_last_frame=True,
        testing=False,
    ):
        """
        Triangulator class for processing 2D keypoint predictions and generating 3D positions.

        Parameters:
        predictions_2d_directory (str): Directory where 2D predictions are stored for this video.
        output_directory_triangulation (str): Directory to save resulting memmap files.
        camera_sync_file (str): CSV file with camera sync data.
        expected_frames_per_video (int): Expected number of frames per video.
        camera_calibration_directory (str): Directory with calibration data in jarvis format.
        n_jobs (int): Number of parallel jobs. Default is 5.
        keep_top_k (int): Number of top keypoints to keep. Default is 3.
        perform_top_k_filtering (bool): Flag to perform top k filtering. Default is False.
        leave_one_out_center_threshold_mm (float): Threshold distance for leave-one-out center in mm. Default is 50.
        threshold_distance_from_parent_mm (float): Threshold distance from parent keypoint in mm. Default is 50.
        mmap_dtype (str): Data type for memory-mapped files. Default is "float32".
        print_nans (bool): Flag to print NaNs. Default is True.
        mean_filt_samples (int): Number of samples for mean filtering (~100ms at 120fps). Default is 11.
        mean_filt_distance_thresh_px (float): Distance threshold for mean filtering in pixels. Default is 150.
        ignore_completed (bool): Flag to ignore completed tasks. Default is False.
        suppress_assertion (bool): Flag to suppress assertion. Default is False.
        sub_last_frame (bool): Flag to subtract last frame. Default is True.
        testing (bool): Flag for testing. Will reduce number of frames used to 5000. Default is False.
        """
        self.predictions_2d_directory = Path(predictions_2d_directory)
        self.output_directory_triangulation = Path(output_directory_triangulation)
        self.camera_sync_file = Path(camera_sync_file)
        self.camera_calibration_directory = Path(camera_calibration_directory)
        self.expected_frames_per_video = int(expected_frames_per_video)
        self.n_jobs = n_jobs
        self.keep_top_k = keep_top_k
        self.perform_top_k_filtering = perform_top_k_filtering
        self.leave_one_out_center_threshold_mm = leave_one_out_center_threshold_mm
        self.threshold_distance_from_parent_mm = threshold_distance_from_parent_mm
        self.mmap_dtype = mmap_dtype
        self.print_nans = print_nans
        self.mean_filt_samples = mean_filt_samples
        self.mean_filt_distance_thresh_px = mean_filt_distance_thresh_px
        self.recompute_completed = recompute_completed
        self.suppress_assertion = suppress_assertion
        self.sub_last_frame = sub_last_frame
        self.testing = testing

        # Initialize keypoint and skeleton information from dataset_info
        keypoint_info = dataset_info["keypoint_info"]
        skeleton_info = dataset_info["skeleton_info"]
        self.keypoint_idx = {value["name"]: key for key, value in keypoint_info.items()}

        # Order of keypoints as predicted
        self.keypoints = np.array([keypoint_info[i]["name"] for i in keypoint_info.keys()])
        self.n_keypoints = len(self.keypoints)

    def check_if_triangulation_exists(self):
        # checks a log saved in output_directory_triangulation to see if the triangulation has already been completed
        if (self.output_directory_triangulation / "triangulation_completed.log").exists():
            return True
        else:
            return False

    def save_triangulation_completed(self):
        with open(self.output_directory_triangulation / "triangulation_completed.log", "w") as f:
            f.write("Triangulation completed")

    def run(self):

        # check if the triangulation has already been completed
        if not self.recompute_completed:
            if self.check_if_triangulation_exists():
                logger.info("Triangulation already exists")
                return

        # ensure the camera sync already exists
        if not self.camera_sync_file.exists():
            raise FileNotFoundError("Camera sync data not found")

        # load the camera sync file
        #  there currently is one -1 at the end of each camera sync file
        #  if that gets fixed, we can remove the [:-1]
        camera_sync_df = pd.read_csv(self.camera_sync_file)
        self.n_frames = len(camera_sync_df)
        if self.sub_last_frame:
            self.n_frames -= 1

        # compute the number of expected videos
        n_videos_expected = round(np.ceil(self.n_frames / self.expected_frames_per_video))
        logger.info(f"\t n_videos_expected {n_videos_expected}")

        self.all_extrinsics, self.all_intrinsics, camera_names = mcc.load_calibration(
            self.camera_calibration_directory,
            load_format="jarvis",
        )
        self.cameras = camera_names
        self.n_cameras = len(self.cameras)
        logging.info(f"\t n_cameras {self.n_cameras}")

        assert self.n_cameras > 0, "No cameras found"

        # load the 2D predictions
        self.load_predictions()

        # ensure that we have a complete dataset
        if self.suppress_assertion == False:
            assert (
                int(len(self.recording_predictions) / self.n_cameras) == n_videos_expected
            ), f"Expected {n_videos_expected} videos, got {int(len(self.recording_predictions) / self.n_cameras)}"

        # Create a temporary directory
        with tempfile.TemporaryDirectory() as tmpdirname:
            # Convert the temporary directory path to a Path object
            tmpdir_path = Path(tmpdirname)
            # prepare the output files
            (
                tmp_predictions_2d_file,
                tmp_predictions_reproj_2d_file,
                tmp_predictions_raw_2d_file,
                tmp_confidences_2d_file,
                tmp_predictions_3d_file,
                tmp_confidences_3d_file,
                tmp_reprojection_errors_file,
            ) = self.create_output_files(tmpdir_path)

            # NOTE: chunks are based on the length of videos, so if a video is e.g. 120 minutes long
            #  this can be a large memory requirement
            logger.info("Running triangulation over chunks")
            for chunk_i, start_frame in enumerate(
                tqdm(np.sort(self.recording_predictions.frame.unique()), desc="chunk")
            ):
                self.triangulate_chunk(start_frame, chunk_i)

            # Check if the destination file exists and remove it if it does
            def move_and_overwrite(src, dst_dir):
                dst = os.path.join(dst_dir, os.path.basename(src))
                if os.path.exists(dst):
                    os.remove(dst)
                shutil.move(src, dst)

            # move the final temp_coordinates_file to the output directory
            move_and_overwrite(tmp_predictions_2d_file, self.output_directory_triangulation)
            move_and_overwrite(tmp_predictions_reproj_2d_file, self.output_directory_triangulation)
            move_and_overwrite(tmp_predictions_raw_2d_file, self.output_directory_triangulation)
            move_and_overwrite(tmp_confidences_2d_file, self.output_directory_triangulation)
            move_and_overwrite(tmp_predictions_3d_file, self.output_directory_triangulation)
            move_and_overwrite(tmp_confidences_3d_file, self.output_directory_triangulation)
            move_and_overwrite(tmp_reprojection_errors_file, self.output_directory_triangulation)

        # mark triangulation as completed
        logger.info("Saving triangulations")
        self.save_triangulation_completed()

    def triangulate_chunk(self, start_frame, chunk_i):


        if not self.testing:
            # Load in one file to determine how many frames in this chunk
            # and check that all cameras have the same number of frames for this chunk.
            nframes_per_camera_chunk = []
            for ci, camera in enumerate(self.cameras):
                camera_2d_row = self.recording_predictions[
                    (self.recording_predictions.camera == camera)
                    & (self.recording_predictions.frame == start_frame)
                ].iloc[0]
                with h5py.File(camera_2d_row.file, "r") as h5f:
                    keypoint_coords = np.array(h5f["keypoint_coords"])
                nframes_per_camera_chunk.append(keypoint_coords.shape[0])
            assert len(set(nframes_per_camera_chunk)) == 1, "Not all cameras have the same number of frames for chunk {chunk_i}"
            n_frames_chunk = nframes_per_camera_chunk[0]
        else:
            logging.info("Running in testing mode, using only first 5000 frames.")
            n_frames_chunk = 5000

        # determine the end frame
        # end_frame = np.min([self.n_frames, start_frame + self.expected_frames_per_video])
        chunk_start = int(start_frame)
        chunk_end = int(chunk_start + n_frames_chunk)

        # for each camera, populate the confidences and positions from hdf5
        confidences_2d_chunk = np.zeros(
            (chunk_end - chunk_start, len(self.cameras), self.n_keypoints)
        )
        confidences_2d_chunk[:] = np.nan
        positions_2d_chunk = (
            np.zeros((chunk_end - chunk_start, len(self.cameras), self.n_keypoints, 2)) * np.nan
        )  # nframes, ncameras, nkeypoints, xy
        positions_reprojected_2d_chunk = (
            np.zeros((chunk_end - chunk_start, len(self.cameras), self.n_keypoints, 2)) * np.nan
        )
        positions_2d_chunk[:] = np.nan
        positions_reprojected_2d_chunk[:] = np.nan
        detection_conf_chunk = np.zeros((chunk_end - chunk_start, len(self.cameras))) * np.nan
        detection_conf_chunk[:] = np.nan
        detection_coords_chunk = np.zeros((chunk_end - chunk_start, len(self.cameras), 4)) * np.nan
        detection_coords_chunk[:] = np.nan

        for ci, camera in enumerate(self.cameras):
            camera_2d_row = self.recording_predictions[
                (self.recording_predictions.camera == camera)
                & (self.recording_predictions.frame == start_frame)
            ].iloc[0]
            with h5py.File(camera_2d_row.file, "r") as h5f:
                keypoint_coords = np.array(h5f["keypoint_coords"])
                keypoint_conf = np.array(h5f["keypoint_conf"])
                detection_conf = np.array(h5f["detection_conf"])
                detection_coords = np.array(h5f["detection_coords"])

            chunk_slice = slice(chunk_start, chunk_end)
            positions_2d_chunk[:, ci] = np.squeeze(keypoint_coords)[chunk_slice]
            confidences_2d_chunk[:, ci] = np.squeeze(keypoint_conf)[chunk_slice]
            confidences_2d_chunk_copy = confidences_2d_chunk.copy()[chunk_slice]
            detection_coords_chunk[:, ci] = np.squeeze(detection_coords)[chunk_slice]
            detection_conf_chunk[:, ci] = np.squeeze(detection_conf)[chunk_slice]
        if self.print_nans:
            prop_nan = np.mean(np.isnan(positions_2d_chunk))
            logger.info(f"\tInitial prop. 2D keypoints are NaNs: {round(prop_nan, 3)}")

        # Save a copy of the raw 2D positions (helps in gimbal later)
        positions_2d_chunk_raw = positions_2d_chunk.copy()

        # remove keypoint detections that are below confidence threshold for each camera
        keypoint_names = [keypoint_info[i]["name"] for i in keypoint_info.keys()]
        for ci, camera in enumerate(self.cameras):
            cam_type = "side" if "side" in camera else camera
            conf_thresholds = conf_thresholds_by_camera[cam_type]
            for ki, keypoint in enumerate(keypoint_names):
                this_thresh = [val for keypoint_type,val in conf_thresholds.items() if keypoint_type in keypoint]
                assert len(this_thresh) == 1, f"Expected 1 threshold, got {len(this_thresh)}"
                this_thresh = this_thresh[0]
                these_confs = confidences_2d_chunk[:, ci, ki]
                confidences_2d_chunk_copy[:, ci, ki][these_confs < this_thresh] = np.nan
                positions_2d_chunk[:, ci, ki][these_confs < this_thresh] = np.nan
        if self.print_nans:
            prop_nan = np.mean(np.isnan(positions_2d_chunk))
            logger.info(f"\t Prop. 2D keypoint NaNs after conf thresholding: {round(prop_nan, 3)}")

        # JP: this is mostly excluding perfectly good frames, so removing from my pipeline.
        # Remove entire frames where the keypoint-centroid isn't aligned with the other cameras
        # positions_2d_chunk, confidences_2d_chunk = leave_one_out_2d_filter(
        #     positions_2d_chunk,
        #     self.cameras,
        #     detection_coords_chunk,
        #     self.all_extrinsics,
        #     self.all_intrinsics,
        #     confidences_2d_chunk,
        #     leave_one_out_center_threshold=self.leave_one_out_center_threshold_mm,
        #     n_jobs=self.n_jobs,
        #     plot_results=False,
        #     verbose=self.print_nans,
        # )

        # if self.print_nans:
        #     prop_nan = np.mean(np.isnan(positions_2d_chunk))
        #     logger.info(f"\tProp 2D keypoint NaNs after LOO 2D: {round(prop_nan, 3)}")

        # compute distance of each point to the median over n samples
        mean_kpt_movement = uniform_filter(
            np.nan_to_num(
                np.sqrt(np.sum(np.diff(positions_2d_chunk, axis=0) ** 2, axis=(-1, -2)))
            ),
            self.mean_filt_samples,
            axes=0,
        )
        m = mean_kpt_movement > self.mean_filt_distance_thresh_px
        positions_2d_chunk[np.concatenate([m, m[-1:]])] = np.nan
        positions_2d_chunk[np.concatenate([m[:1], m])] = np.nan
        if self.print_nans:
            prop_nan = np.mean(np.isnan(positions_2d_chunk))
            logger.info(f"\tProp 2D keypoint NaNs after movement distance filter: {round(prop_nan, 3)}")

        # only keep the top k confidence cameras
        #    generally not needed, since we're weighting by confidence in triangulation
        if self.perform_top_k_filtering:
            positions_2d_chunk = filter_top_k(
                positions_2d_chunk, confidences_2d_chunk, k=self.keep_top_k
            )

        # set 2d conf copy to nan anywhere the 2d pred is also nan,
        # so that we can extract the 3d confidences from the 2d confidences that were actually used.
        confidences_2d_chunk_copy[np.isnan(positions_2d_chunk[...,0])] = np.nan

        # compute position in 3D
        positions_3D_chunk = Parallel(n_jobs=self.n_jobs)(
            delayed(mcc.triangulate)(
                positions_2d_chunk[frame_num, :, :, :],
                self.all_extrinsics,
                self.all_intrinsics,
                # confidences_2d_chunk[frame_num],
            )
            for frame_num in tqdm(
                range(len(positions_2d_chunk)), desc="3D triangulation", leave=False
            )
        )
        positions_3D_chunk = np.stack(positions_3D_chunk)

        if self.print_nans:
            prop_nan = np.mean(np.isnan(positions_3D_chunk))
            print(f"\tProp 3D keypoint NaNs after triangulation: {round(prop_nan, 3)}")

        # filter keypoints that are very far from their parent
        positions_3D_chunk = filter_3d_keypoints_based_on_distance_from_parent(
            positions_3D_chunk,
            parents_dict,
            keypoints=self.keypoints,
            threshold_distance_from_parent=self.threshold_distance_from_parent_mm,
        )

        if self.print_nans:
            prop_nan = np.mean(np.isnan(positions_3D_chunk))
            print(f"\Prop 3D keypoint NaNs after 3d filter: {round(prop_nan, 3)}")

        # get reprojections
        positions_2D_reprojections = np.zeros(positions_2d_chunk.shape) * np.nan
        for cam in range(len(self.cameras)):
            extrinsics = self.all_extrinsics[cam]
            camera_matrix, dist_coefs = self.all_intrinsics[cam]
            positions_2D_reprojections[:, cam, :, :] = mcc.project_points(
                positions_3D_chunk,
                extrinsics=extrinsics,
                camera_matrix=camera_matrix,
                dist_coefs=dist_coefs,
            )
        if self.print_nans:
            prop_nan = np.mean(np.isnan(positions_3D_chunk))
            print(f"\tTotal: Prop 3D keypoints are NaNs: {round(prop_nan, 3)}")

        # compute the reprojection errors for each keypoint / camera
        reprojection_errors_chunk = np.sqrt(
            ((positions_2d_chunk - positions_2D_reprojections) ** 2).sum(axis=-1)
        )

        # populate mmaps
        self.reprojection_errors[chunk_start:chunk_end] = reprojection_errors_chunk
        self.keypoints_2d[chunk_start:chunk_end] = positions_2d_chunk
        self.keypoints_reproj_2d[chunk_start:chunk_end] = positions_2D_reprojections
        self.keypoints_raw_2d[chunk_start:chunk_end] = positions_2d_chunk_raw
        self.confidences_2d[chunk_start:chunk_end] = confidences_2d_chunk
        self.keypoints_3d[chunk_start:chunk_end] = positions_3D_chunk
        self.confidences_3d[chunk_start:chunk_end] = np.nanmedian(confidences_2d_chunk_copy, axis=1)  # extract 3D confidences only from the 2D confidences that were actually used

        # plot reprojection errors
        median_reprojection_errors = np.nanmedian(reprojection_errors_chunk, axis=0)
        fig, ax = plt.subplots(figsize=(10, 2))
        mat = ax.matshow(median_reprojection_errors)
        plt.colorbar(mat)
        ax.set_xticks(range(len(self.keypoints)), self.keypoints, rotation=90)
        ax.set_yticks(range(len(self.cameras)), self.cameras)
        # save to file for chunk
        reprojection_errors_file = (
            self.output_directory_triangulation / f"reprojection_errors_{chunk_i}.jpg"
        )
        plt.savefig(reprojection_errors_file)
        plt.close()

        # plot sample
        X = positions_3D_chunk[:1500, :, 0]
        fig, ax = plt.subplots(figsize=(10, 1))
        ax.matshow(X.T - np.mean(X, axis=1).T, aspect="auto")
        triangulation_sample_file = (
            self.output_directory_triangulation / f"triangulation_sample_{chunk_i}.jpg"
        )
        plt.savefig(triangulation_sample_file)
        plt.close()

    def create_output_files(self, tmpdir_path):
        logger.info("Creating output files")
        # set the shape and dtype of arrays
        keypoints_2d_shape = (self.n_frames, self.n_cameras, self.n_keypoints, 2)
        keypoints_2d_dtype = self.mmap_dtype
        keypoints_2d_shape_str = "x".join(map(str, keypoints_2d_shape))

        confidences_2d_shape = (self.n_frames, self.n_cameras, self.n_keypoints)
        confidences_2d_dtype = self.mmap_dtype
        confidences_2d_shape_str = "x".join(map(str, confidences_2d_shape))

        keypoints_3d_shape = (self.n_frames, self.n_keypoints, 3)
        keypoints_3d_dtype = self.mmap_dtype
        keypoints_3d_shape_str = "x".join(map(str, keypoints_3d_shape))

        confidences_3d_shape = (self.n_frames, self.n_keypoints)
        confidences_3d_dtype = self.mmap_dtype
        confidences_3d_shape_str = "x".join(map(str, confidences_3d_shape))

        reprojection_errors_shape = (self.n_frames, self.n_cameras, self.n_keypoints)
        reprojection_errors_dtype = self.mmap_dtype
        reprojection_errors_shape_str = "x".join(map(str, reprojection_errors_shape))

        predictions_3d_file = (
            tmpdir_path / f"predictions_3d.{keypoints_3d_dtype}.{keypoints_3d_shape_str}.mmap"
        )
        predictions_2d_file = (
            tmpdir_path / f"predictions_2d.{keypoints_2d_dtype}.{keypoints_2d_shape_str}.mmap"
        )
        predictions_reproj_2d_file = (
            tmpdir_path / f"predictions_reproj_2d.{keypoints_2d_dtype}.{keypoints_2d_shape_str}.mmap"
        )
        predictions_raw_2d_file = (
            tmpdir_path / f"predictions_raw_2d.{keypoints_2d_dtype}.{keypoints_2d_shape_str}.mmap"
        )
        confidences_2d_file = (
            tmpdir_path / f"confidences_2d.{confidences_2d_dtype}.{confidences_2d_shape_str}.mmap"
        )
        confidences_3d_file = (
            tmpdir_path / f"confidences_3d.{confidences_3d_dtype}.{confidences_3d_shape_str}.mmap"
        )
        reprojection_errors_file = (
            tmpdir_path
            / f"reprojection_errors.{reprojection_errors_dtype}.{reprojection_errors_shape_str}.mmap"
        )

        # create memmap to write to
        self.keypoints_2d = np.memmap(
            predictions_2d_file,
            dtype=keypoints_2d_dtype,
            mode="w+",
            shape=keypoints_2d_shape,
        )

        self.keypoints_reproj_2d = np.memmap(
            predictions_reproj_2d_file,
            dtype=keypoints_2d_dtype,
            mode="w+",
            shape=keypoints_2d_shape,
        )

        self.keypoints_raw_2d = np.memmap(
            predictions_raw_2d_file,
            dtype=keypoints_2d_dtype,
            mode="w+",
            shape=keypoints_2d_shape,
        )

        self.confidences_2d = np.memmap(
            confidences_2d_file,
            dtype=np.float32,
            mode="w+",
            shape=confidences_2d_shape,
        )

        self.keypoints_3d = np.memmap(
            predictions_3d_file,
            dtype=keypoints_3d_dtype,
            mode="w+",
            shape=keypoints_3d_shape,
        )

        self.confidences_3d = np.memmap(
            confidences_3d_file,
            dtype=confidences_3d_dtype,
            mode="w+",
            shape=confidences_3d_shape,
        )

        self.reprojection_errors = np.memmap(
            reprojection_errors_file,
            dtype=reprojection_errors_dtype,
            mode="w+",
            shape=reprojection_errors_shape,
        )

        return (
            predictions_2d_file,
            predictions_reproj_2d_file,
            predictions_raw_2d_file,
            confidences_2d_file,
            predictions_3d_file,
            confidences_3d_file,
            reprojection_errors_file,
        )

    def load_predictions(self):
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
        self.recording_predictions = self.recording_predictions[
            self.recording_predictions.camera.isin(self.cameras)
        ]
        # assert that there is the same number of frames for each camera
        assert same_frames_for_all_cameras(self.recording_predictions)


def filter_3d_keypoints_based_on_distance_from_parent(
    positions_3D, parents_dict, keypoints, threshold_distance_from_parent=50
):
    """
    Filters 3D keypoints based on their distance from parent keypoints.

    Parameters:
    positions_3D (np.ndarray): 3D positions of keypoints, shape (N, M, 3),
                               where N is the number of frames, M is the number of keypoints.
    parents_dict (dict): Dictionary where keys are keypoints and values are their parent keypoints.
    keypoints (list): List of keypoints.
    threshold_distance_from_parent (float): Distance threshold in mm to consider a keypoint as an outlier.

    Returns:
    np.ndarray: Filtered 3D positions with outliers set to np.nan.
    """
    # Get the indices of parent keypoints for each keypoint
    parents_idx = np.array(
        [np.where(parents_dict[i] == np.array(keypoints))[0][0] for i in keypoints]
    )

    # Calculate the distances of keypoints from their parents
    keypoint_distances_from_parent = np.sqrt(
        np.sum(
            (positions_3D - positions_3D[:, parents_idx]) ** 2,
            axis=(2),
        )
    )

    # Identify keypoints with distances greater than the threshold
    outliers = keypoint_distances_from_parent > threshold_distance_from_parent

    # Set outlier positions to np.nan
    positions_3D[outliers] = np.nan

    return positions_3D


def filter_top_k(keypoint_data, confidence_data, k=3):
    """
    Filters the top k keypoints based on confidence data.

    Parameters:
    keypoint_data (np.ndarray): 3D positions of keypoints, shape (N, M, 3).
    confidence_data (np.ndarray): Confidence scores of keypoints, shape (N, M, 3).
    k (int): Number of keypoints to filter based on confidence.

    Returns:
    np.ndarray: Filtered keypoint data with top k keypoints set to np.nan.
    """
    # Get the indices of the top k keypoints based on confidence scores
    top_k_indices = np.argsort(confidence_data, axis=1)[:, :k, :]

    # Set the positions of the top k keypoints to np.nan
    for j in range(top_k_indices.shape[2]):
        keypoint_data[np.arange(top_k_indices.shape[0])[:, None], top_k_indices[:, :, j], j, :] = (
            np.nan
        )

    return keypoint_data


def same_frames_for_all_cameras(df):
    """
    Checks if all cameras have the same set of frames.

    Parameters:
    df (pd.DataFrame): DataFrame containing camera and frame columns.

    Returns:
    bool: True if all cameras have the same set of frames, False otherwise.
    """
    # Group by camera and get the set of frames for each camera
    camera_frames = df.groupby("camera")["frame"].apply(set)

    # Check if all cameras have the same set of frames
    if len(camera_frames.apply(frozenset).unique()) == 1:
        return True
    else:
        # Print the cameras with different frames
        logger.info("Different frames for cameras")
        for camera, frames in camera_frames.items():
            logger.info(f"{camera}: {frames}")
        return False


def leave_one_out_2d_filter(
    positions_2D,
    cameras,
    detection_coordinates,
    all_extrinsics,
    all_intrinsics,
    confidences,
    leave_one_out_center_threshold=50,
    n_jobs=-1,
    plot_results=False,
    verbose=False,
):
    """
    Perform leave-one-out filtering on 2D keypoint detections to identify and filter out erroneous detections.

    Parameters:
    positions_2D (np.ndarray): Array of 2D keypoint positions, shape (frames, cameras, keypoints, 2).
    cameras (list): List of camera identifiers.
    detection_coordinates (np.ndarray): Array of detection coordinates.
    all_extrinsics (list): List of extrinsic parameters for each camera.
    all_intrinsics (list): List of intrinsic parameters for each camera.
    confidences (np.ndarray): Array of confidence scores for detections, shape (frames, cameras, keypoints).
    leave_one_out_center_threshold (float): Distance threshold for considering a detection as erroneous.
    n_jobs (int): Number of jobs for parallel processing.
    plot_results (bool): Flag to plot results (not implemented).

    Returns:
    tuple: Filtered positions_2D and confidences.
    """
    # points where a new detection is performed
    # detection_changes = np.concatenate(
    #     [
    #         [0],
    #         np.where(np.sum(np.diff(detection_coordinates, axis=0), axis=(1, 2)) > 0)[0],
    #         [len(detection_coordinates) - 1],
    #     ]
    # )

    detection_changes = np.arange(0, len(detection_coordinates))

    for cami in tqdm(range(len(cameras)), leave=False, desc="camera"):
        other_cams = np.array([i for i in np.arange(len(cameras)) if i != cami])
        # triangulate 3D positions leaving out the main camera
        LOO_detections = Parallel(n_jobs=n_jobs)(
            delayed(mcc.triangulate)(
                positions_2D[frame_num, other_cams, :],
                [all_extrinsics[i] for i in other_cams],
                [all_intrinsics[i] for i in other_cams],
                # confidences[frame_num, other_cams],
            )
            for frame_num in tqdm(detection_changes, desc="L.O.O. 3D triangulation", leave=False)
        )
        # project back into the left out camera
        # get reprojections
        extrinsics = all_extrinsics[cami]
        camera_matrix, dist_coefs = all_intrinsics[cami]
        LOO_positions_2D_reprojections = mcc.project_points(
            np.array(LOO_detections),
            extrinsics=extrinsics,
            camera_matrix=camera_matrix,
            dist_coefs=dist_coefs,
        )

        import pdb; pdb.set_trace()


        # Calculate median of the reprojections and original predictions
        LOO_prediction_center_2d = np.nanmedian(LOO_positions_2D_reprojections, axis=1)
        original_prediction_center_2d = np.nanmedian(positions_2D[detection_changes, cami], axis=1)

        # Calculate distances between LOO predictions and original predictions
        LOO_original_prediction_distances = np.sqrt(
            np.sum(
                (LOO_prediction_center_2d - original_prediction_center_2d) ** 2,
                axis=1,
            )
        )
        
        # Identify bad detections based on threshold
        bad_detections = np.where(
            (LOO_original_prediction_distances > leave_one_out_center_threshold)
            | (np.isnan(LOO_original_prediction_distances))
        )[0]

        # Set bad detections to nan and confidences to a very low value
        if verbose:
            logger.info(f"Removing {len(bad_detections)} bad detections via centroid LOO from camera {cameras[cami]}")
        for bad_detection_idx in bad_detections:
            if bad_detection_idx + 1 == len(detection_changes):
                continue
            idx_start = detection_changes[bad_detection_idx]
            idx_end = detection_changes[bad_detection_idx + 1]
            # set keypoints to nan
            positions_2D[idx_start:idx_end, cami] = np.nan
            # set confidence to zero
            confidences[idx_start:idx_end, cami] = 1e-10

    return positions_2D, confidences


def generate_initial_positions(positions):
    """
    Generate initial positions by interpolating missing values in a 3D array.

    Parameters:
    positions (np.ndarray): Array of positions, shape (frames, keypoints, coordinates).

    Returns:
    np.ndarray: Interpolated positions, same shape as input.
    """
    # Initialize an array of zeros with the same shape as positions
    init_positions = np.zeros_like(positions)

    # Iterate over each keypoint
    for k in range(positions.shape[1]):
        # Find indices of non-NaN values for the current keypoint
        ix = np.nonzero(~np.isnan(positions[:, k, 0]))[0]

        # Iterate over each coordinate (e.g., x, y, z)
        for i in range(positions.shape[2]):
            # Interpolate missing values for the current coordinate of the keypoint
            init_positions[:, k, i] = np.interp(
                np.arange(positions.shape[0]), ix, positions[:, k, i][ix]
            )

    return init_positions
