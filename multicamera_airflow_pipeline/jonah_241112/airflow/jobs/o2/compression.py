from datetime import datetime
import logging
from pathlib import Path
import textwrap
import time

import av
import yaml

from multicamera_airflow_pipeline.jonah_241112.interface.o2 import O2Runner
from multicamera_airflow_pipeline.utils.naming_utils import split_multicam_filename

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

def get_video_len(vid):
    c = av.open(Path(vid).as_posix())
    return c.streams.video[0].frames


def convert_minutes_to_hms(minutes_float):
    # Convert minutes to total seconds
    total_seconds = int(minutes_float * 60)

    # Extract hours, minutes, and seconds using divmod
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    # Format as HH:MM:SS
    return f"{hours:02}:{minutes:02}:{seconds:02}"


def check_compression_completion(output_directory_log, all_videos):
    completed = {}
    for video in all_videos:
        cam = split_multicam_filename(video.name)["camera"]
        if not (output_directory_log / f"completed_{cam}.txt").exists():
            completed[video] = False
        else:
            completed[video] = True
        
    return completed


def compression(
    recording_row,
    job_directory,
    output_directory,
    config_file,
):
    # load config
    config_file = Path(config_file)
    config = yaml.safe_load(open(config_file, "r"))

    # where the video data is located
    recording_directory = (
        Path(recording_row.video_location_on_o2) / recording_row.video_recording_id
    )

    assert (
        recording_directory.exists()
    ), f"Recording directory {recording_directory} does not exist"

    # where to save output
    output_directory_log = (
        output_directory / "compression" / recording_row.video_recording_id
    )
    output_directory_log.mkdir(parents=True, exist_ok=True)
    current_datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    remote_job_directory = job_directory / "compression" / f"{recording_row.video_recording_id}_{current_datetime_str}"

    # Find all videos to compress
    all_videos = list(recording_directory.glob("*.mp4"))
    all_videos = [v for v in all_videos if not any([p in v.name for p in config["compression"]["patterns_to_exclude_from_vids"]])]
    logger.info(f"Found {len(all_videos)} videos")
    assert len(all_videos) > 0, f"No videos found in {recording_directory}"

    # check if airflow task is already successfully completed
    # from multicamera_airflow_pipeline.jonah_241112.airflow.dag_o2 import dummy_dag
    # downstream_tasks = dummy_dag.get_all_downstream_tasks(recording_row.overwrite_from) | set([recording_row.overwrite_from])
    # if not recording_row.overwrite or (recording_row.overwrite and ("compression" not in downstream_tasks)):
    #     completed_dict = check_compression_completion(output_directory_log, all_videos)
    #     if all(completed_dict.values()):
    #         logger.info("Compression completed, quitting")
    #         return
    #     else:
    #         logger.info("Compression incomplete, starting")

    params = {
        "recompute_completed":recording_row.overwrite,
        # "recording_directory": recording_directory.as_posix(),
        "output_directory_log": output_directory_log.as_posix(),
    }

    duration_requested = convert_minutes_to_hms(
        max([
            recording_row.duration_m * config["o2"]["compression"]["o2_runtime_multiplier"],
            15,
        ])
    )
    
    runners = {}
    for video in all_videos:
        cam = split_multicam_filename(video.name)["camera"]

        # create the job runner
        runner = O2Runner(
            job_name_prefix=f"{recording_row.video_recording_id}_{cam}_compression",
            remote_job_directory=remote_job_directory,
            conda_env=config["o2"]["compression"]["conda_env"],
            o2_username=recording_row.username,
            o2_server="login.o2.rc.hms.harvard.edu",
            job_params=params,
            o2_n_cpus=config["o2"]["compression"]["o2_n_cpus"],
            o2_memory=config["o2"]["compression"]["o2_memory"],
            o2_time_limit=duration_requested,
            o2_queue=config["o2"]["compression"]["o2_queue"],
            modules_to_load=["gcc/9.2.0"],
        )
    
        runner.python_script = textwrap.dedent(
            f"""
            # load params
            import yaml
            params_file = "{runner.remote_job_directory / f"{runner.job_name}.params.yaml"}"
            config_file = "{config_file.as_posix()}"

            params = yaml.safe_load(open(params_file, 'r'))
            config = yaml.safe_load(open(config_file, 'r'))

            # grab func
            from multicamera_airflow_pipeline.jonah_241112.compression import VideoCompressor
            compressor = VideoCompressor(
                "{video}",
                **params,
                **config["compression"],
            )
            compressor.run()
            """
        )

        print(runner.python_script)

        runner.run()
        runners[cam] = runner

    # wait until the job is finished
    # 10000/60/24 = roughly 1 week
    for i in range(10000):
        # check job status every n seconds
        status = all([runner.check_job_status() for runner in runners.values()]) 
        if status:
            break
        time.sleep(60)

    # check if sync successfully completed
    completed_dict = check_compression_completion(output_directory_log, all_videos)
    if all(completed_dict.values()):
        logger.info("Compression completed successfully")
    else:
        logger.warning("Compression did not complete successfully.")
        logger.info("Removing remaining temporary files...")
        tmp_videos = list(recording_directory.glob("*.tmp.mp4"))
        for tmp_video in tmp_videos:
            tmp_video.unlink()
        raise ValueError("Compression did not complete successfully.")
