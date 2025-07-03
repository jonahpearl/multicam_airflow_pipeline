import re
import sys

import click
import numpy as np
from o2_utils.slurm import evaluate_resource_usage
from tqdm.auto import tqdm
import yaml


@click.command()
@click.argument("airflow_logs_file", type=click.Path(exists=True))
@click.option(
    "--dag-name-regex",
    default=None,
    type=str,
    help="Regex to filter DAG names",
)
@click.option(
    "--min-job-time",
    default=0,
    type=float,
    help="Minimum job time in seconds to consider for analysis",
)
def main(airflow_logs_file, dag_name_regex=None, min_job_time=0):
    """Analyze the jobs in the airflow logs file.

    Args
    ----
    airflow_logs_file : str
        Path to the airflow logs file. (See `scripts/export_logs_from_airflow.py`.)

    dag_name_regex : str, optional
        If provided, only analyze the DAGs whose names match this regex.

    Returns
    -------
    None
    """

    print(f"Analyzing jobs from {airflow_logs_file}...")

    # Read the logs
    with open(airflow_logs_file, "r") as f:
        logs = yaml.safe_load(f)

    tasks_to_analyze = [
        "compression",
        "predict_2d",
        "triangulation",
        "run_gimbal",
        "size_normalization",
        "arena_alignment",
        "egocentric_alignment",
        "compute_continuous_features",
        "validation_videos",
    ]

    # Sub-select DAGs of interest
    if dag_name_regex is not None:
        logs = {
            dag_name: v
            for dag_name, v in logs.items()
            if re.match(dag_name_regex, dag_name)
        }
        print(f"Sub-selected {len(logs)} DAGs: {list(logs.keys())}")

    # See how long each task took relative to the time requested
    for task in tqdm(tasks_to_analyze, desc="Airflow steps"):

        # if task != "arena_alignment": continue

        time_fractions = []
        mem_fractions = []
        jobids = []
        for dag_name, run_info in tqdm(logs.items(), desc="Runs", leave=False):
            # Skip stuff we dont want
            if (
                dag_name == "refresh_pipeline_dag"
                or task not in run_info
                or run_info[task]["job_id"] == "Not found"
            ):
                continue

            # Get the job info
            jobid = run_info[task]["job_id"]
            jobinfo = evaluate_resource_usage(
                jobid, plot=False
            )  # calls squeue in bkgnd

            # If job isn't finished w status COMPLETED, will not be returned in the dict, so skip that
            if jobid not in jobinfo:
                continue
            
            # If job took less than the minimum time, skip it
            if jobinfo[jobid]["run_time"] < min_job_time:
                continue

            # Save the info
            time_fractions.append(jobinfo[jobid]["time_fraction"])
            mem_fractions.append(jobinfo[jobid]["mem_fraction"])
            jobids.append(jobid)

        if len(time_fractions) == 0:
            print(f"No jobs found for task {task} with the given criteria.")
            continue

        # Report the results
        print(f"Task: {task}")
        print(f"\tAnalyzed {len(time_fractions)} jobs")
        print(f"\tJob ids: {jobids}")
        print(f"\tMinimum time fraction: {np.min(time_fractions):.3f}")
        print(f"\tMedian time fraction: {np.median(time_fractions):.3f}")
        print(f"\tMaximum time fraction: {np.max(time_fractions):.3f}")
        print()
        print(f"\tMinimum mem fraction: {np.min(mem_fractions):.3f}")
        print(f"\tMedian mem fraction: {np.median(mem_fractions):.3f}")
        print(f"\tMaximum mem fraction: {np.max(mem_fractions):.3f}")

    # TODO: Plot the results

    return


if __name__ == "__main__":
    main()