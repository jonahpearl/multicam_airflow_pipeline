from datetime import datetime, timedelta
import pandas as pd
import requests
from io import BytesIO
from pathlib import Path
from multicamera_airflow_pipeline.tim_240731.interface.o2 import O2Runner
from datetime import datetime
import textwrap
import inspect
import time
import yaml

import logging

logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger(__name__)


def convert_minutes_to_hms(minutes_float):
    # Convert minutes to total seconds
    total_seconds = int(minutes_float * 60)

    # Extract hours, minutes, and seconds using divmod
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    # Format as HH:MM:SS
    return f"{hours:02}:{minutes:02}:{seconds:02}"


def process_thermistor_data(
    recording_row,
    job_directory,
    output_directory,
    config_file,
):

    # where the video data is located
    recording_directory = (
        Path(recording_row.video_location_on_o2) / recording_row.video_recording_id
    )
    assert (
        recording_directory.exists()
    ), f"Recording directory {recording_directory} does not exist"

    # load config
    config_file = Path(config_file)
    config = yaml.safe_load(open(config_file, "r"))

    # where to save output
    output_directory_thermistor = (
        output_directory / "thermistor_processing" / recording_row.video_recording_id
    )
    logger.info("Creating thermistor processing o2 runner")

    # TODO: could check if already completed, but it's a light analysis and the inner func also checks if it's done.

    output_directory_thermistor.mkdir(parents=True, exist_ok=True)
    current_datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    remote_job_directory = job_directory / current_datetime_str

    # Get params from google sheet
    remove_low_amp_breaths = recording_row.remove_low_amp_breaths
    force_remove_low_amp_breaths = recording_row.force_remove_low_amp_breaths

    # trigger_pin = recording_row.trigzger_pin
    params = {
        "recording_directory": recording_directory.as_posix(),
        "output_directory_thermistor": output_directory_thermistor.as_posix(),
        "remove_low_amp_breaths": remove_low_amp_breaths,
        "force_remove_low_amp_breaths": force_remove_low_amp_breaths,
    }

    # specify the duration of the job (here, it should be short)
    #  in config, set o2_runtime_multiplier to choose how long the job should
    #  run relative to recording duration
    duration_requested = convert_minutes_to_hms(
        recording_row.duration_m * config["o2"]["sync_cameras"]["o2_runtime_multiplier"]
    )

    # create the job runner
    logging.info("Initializing O2 runner")
    runner = O2Runner(
        job_name_prefix=f"{recording_row.video_recording_id}_thermistor_proc",
        remote_job_directory=remote_job_directory,
        conda_env=config["o2"]["thermistor_proc"]["conda_env"],
        o2_username=recording_row.username,
        o2_server="login.o2.rc.hms.harvard.edu",
        job_params=params,
        o2_n_cpus=config["o2"]["thermistor_proc"]["o2_n_cpus"],
        o2_memory=config["o2"]["thermistor_proc"]["o2_memory"],
        o2_time_limit=duration_requested,
        o2_queue=config["o2"]["thermistor_proc"]["o2_queue"],
    )

    runner.python_script = textwrap.dedent(
        f"""
    # load params
    import yaml
    params_file = "{runner.remote_job_directory / f"{runner.job_name}.params.yaml"}"
    config_file = "{config_file.as_posix()}"

    params = yaml.safe_load(open(params_file, 'r'))
    config = yaml.safe_load(open(config_file, 'r'))
        
    # Grab therm proc function
    from multicamera_airflow_pipeline.jonah_240909.sniff.thermistor_proc import ThermistorProcessor
    
    # Init the processor
    processor = ThermistorProcessor(
        recording_directory=params["recording_directory"],
        output_directory=params["output_directory_thermistor"],
        remove_low_amp_breaths=params["remove_low_amp_breaths"],
        force_remove_low_amp_breaths=params["force_remove_low_amp_breaths"],
        **config["thermistor_proc"],
    )
    
    processor.run()
    """
    )
    runner.run()

    # wait until the job is finished
    # 10000/60/24 = roughly 1 week
    for i in range(10000):
        # check job status every n seconds
        status = runner.check_job_status()
        if status:
            break
        time.sleep(60)

    # check if completed
    nwb_filename = output_directory_thermistor.glob("*.nwb")
    if len(list(nwb_filename)) == 0:
        raise FileNotFoundError("No NWB file found in the thermistor processing directory.") 


    validation_dir = output_directory_thermistor.glob("validation*")
    if len(list(validation_dir)) == 0:
        raise FileNotFoundError("No validation directory found in the thermistor processing directory.")
        
    elif len(list(output_directory_thermistor.glob("*.png"))) == 0:
        raise FileNotFoundError("No validation plots found in the thermistor processing directory.")

    logger.info("Thermistor processing completed successfully")