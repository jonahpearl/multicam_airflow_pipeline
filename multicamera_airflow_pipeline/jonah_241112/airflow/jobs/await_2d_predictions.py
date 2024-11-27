from datetime import datetime, timedelta
import logging
from pathlib import Path
import time

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


def await_2d_predictions(
    recording_row,
    output_directory,
    recheck_duration_s=60,
    maximum_wait_time_s=10000,
    patterns_to_exclude_from_vids=["azure", "TRIM"],
):
    """This function waits for 2d predictions to complete, before returning"""

    ### Find the number of videos

    # where the video data is located
    recording_directory = (
        Path(recording_row.video_location_on_o2) / recording_row.video_recording_id
    )
    all_videos = list(recording_directory.glob("*.mp4"))
    # all_videos = [v for v in all_videos if "azure" not in v.stem] #TODO: make this a param
    all_videos = [v for v in all_videos if not any([p in v.name for p in patterns_to_exclude_from_vids])]
    assert len(all_videos) > 0, f"No videos found in {recording_directory}"
    logger.info(f"Found {len(all_videos)} videos")

    output_directory_predictions = (
        output_directory / "2D_predictions" / recording_row.video_recording_id
    )
    all_predictions = list(output_directory_predictions.glob("*.h5"))

    # get the current time
    current_datetime = datetime.now()
    end_time = current_datetime + timedelta(seconds=maximum_wait_time_s)

    completed = False
    while datetime.now() < end_time:
        # check if all predictions are completed
        all_predictions = list(output_directory_predictions.glob("*.h5"))
        if len(all_predictions) == len(all_videos):
            logger.info(f"All predictions completed: {len(all_predictions)}")
            completed=True
            break
        logger.info(
            f"Waiting for {len(all_videos) - len(all_predictions)} predictions to complete"
        )
        # sleep for a bit
        time.sleep(recheck_duration_s)
    logger.info(f"All predictions completed: {len(all_predictions)} / {len(all_videos)}")
    return completed
