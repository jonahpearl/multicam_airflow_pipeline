import logging
import sys

logger = logging.getLogger(__name__)
logger.info(f"Python interpreter binary location: {sys.executable}")

from datetime import datetime
from pathlib import Path
import subprocess
import tempfile
import time

import numpy as np
import paramiko
import yaml


def parse_squeue_output(output):
    """Parse the output of the squeue command into a dict with headers / values.

    """
    lines = output.strip().split('\n')
    headers = lines[0].split()
    job_data = {}
    
    for line in lines[1:]:
        fields = line.split(None, len(headers) - 1)
        job_info = dict(zip(headers, fields))
        job_id = job_info['JOBID']
        job_data[job_id] = job_info
    
    return job_data


class O2Runner:
    """
    A class to run python scripts on O2.
    """

    def __init__(
        self,
        job_name_prefix,
        remote_job_directory,
        conda_env,
        job_params={},
        o2_username="tis697",
        o2_login_server="login.o2.rc.hms.harvard.edu",
        o2_n_cpus=1,
        o2_memory="16G",
        o2_time_limit="4:00:00",
        o2_queue="short",
        modules_to_load=[],
        o2_exclude=None,  # "compute-g-16-175,compute-g-16-176,compute-g-16-177,compute-g-16-194,compute-g-16-197"
        o2_qos=None,  # "gpuquad_qos"
        o2_gres=None,  # "gpu:1"
        do_not_submit=False,
    ):
        self.job_name_prefix = job_name_prefix
        self.remote_job_directory = Path(remote_job_directory)
        self.o2_n_cpus = o2_n_cpus
        self.o2_memory = o2_memory
        self.o2_time_limit = o2_time_limit
        self.o2_queue = o2_queue
        self.o2_username = o2_username
        self.o2_login_server = o2_login_server
        self.conda_env = conda_env
        self.job_params = job_params
        self.o2_exclude = o2_exclude
        self.o2_qos = o2_qos
        self.o2_gres = o2_gres
        self.modules_to_load = modules_to_load
        self.do_not_submit = do_not_submit  # don't actually submit
        self.slurm_job_id = None
        self.ssh_landing_hostname = None

        # determine a job id as the current timestamp
        self.job_datetime = datetime.now()
        self.job_datetime_str = self.job_datetime.strftime("%y-%m-%d-%G-%M-%S-%f")
        self.job_name = f"{self.job_name_prefix}_{self.job_datetime_str}"

        # where the script will be written on the remote server
        self.slurm_script_loc = self.remote_job_directory / f"{self.job_name}.sh"
        self.python_script_loc = self.remote_job_directory / f"{self.job_name}.py"
        self.params_loc = self.remote_job_directory / f"{self.job_name}.params.yaml"
        self.output_log = self.remote_job_directory / f"{self.job_name}_jobid_%j.log"
        self.ssh = None

        self.establish_ssh_connection()

    def report_output_log(self):
        # read the output log and write it to the logger

        # connect to ssh if needed
        if self.ssh is None:
            self.establish_ssh_connection()

        # grab the file if needed
        if not self.output_log.exists():
            self.copy_file_from_remote(
                remote_path=self.output_log,
                local_path=self.output_log,
            )

        # if we still dont have it for some reason, let the user know
        if not self.output_log.exists():
            logger.info(f"(Output log file not found to report error traceback: {self.output_log})")
            return
        else:
            with self.output_log.open("r") as f:
                for line in f:
                    logger.info(line.strip())

    def run(self):

        # check in case a job is already running with this name
        # running_jobs_info = self.get_running_jobs_info_from_o2()
        # for job in running_jobs_info:
        #     if job["NAME"] == self.job_name:


        # create the remote job directory
        logger.info(f"Creating remote job directory: {self.remote_job_directory}")
        self.create_folder_on_remote(self.remote_job_directory)

        logger.info(f"Writing job files to remote directory: {self.remote_job_directory}")
        # write the python script
        self.write_python_script()
        # write the slurm script
        self.write_slurm_script()
        # write the params file
        self.write_params_file()
        # submit the job
        logger.info(f"Submitting job: {self.job_name}")
        if self.do_not_submit:
            logger.info("do_not_submit is True, not submitting job.")
        else:
            self.submit()

        self.close_ssh_connection()

    def write_slurm_script(self):
        slurm_script = "#!/usr/bin/env bash\n"
        slurm_script += f"#SBATCH --partition={self.o2_queue}\n"
        slurm_script += f"#SBATCH --job-name={self.job_name}\n"
        slurm_script += f"#SBATCH --cpus-per-task={self.o2_n_cpus}\n"
        slurm_script += f"#SBATCH --mem={self.o2_memory}\n"
        slurm_script += f"#SBATCH --time={self.o2_time_limit}\n"
        slurm_script += f"#SBATCH --output={self.output_log}\n\n"
        if self.o2_exclude is not None:
            slurm_script += f"#SBATCH --exclude={self.o2_exclude}\n"
        if self.o2_qos is not None:
            slurm_script += f"#SBATCH --qos={self.o2_qos}\n"
        if self.o2_gres is not None:
            slurm_script += f"#SBATCH --gres={self.o2_gres}\n"
        slurm_script += "# Load the required modules\n"
        # slurm_script += f"module load gcc/9.2.0\n\n"
        for modules_to_load in self.modules_to_load:
            slurm_script += f"module load {modules_to_load}\n"
        slurm_script += "source /n/groups/datta/Jonah/miniconda3/etc/profile.d/conda.sh\n"  # load conda
        slurm_script += f"conda activate {self.conda_env}\n\n"
        # slurm_script += f"source activate {self.conda_env}\n\n"
        slurm_script += f"python {self.python_script_loc}\n"

        # save the script to tmp, then move it to the correct location remotely
        with tempfile.NamedTemporaryFile("w", delete=False) as f:
            f.write(slurm_script)
            f.flush()
            self.copy_file_to_remote(local_path=f.name, remote_path=self.slurm_script_loc)

    def write_params_file(self):
        with tempfile.NamedTemporaryFile("w", delete=False) as f:
            # save the params dict to a YAML file
            yaml.dump(self.job_params, f)
            self.copy_file_to_remote(local_path=f.name, remote_path=self.params_loc)

    def write_python_script(self):

        # save the script locally, then move it to the correct location remotely
        with tempfile.NamedTemporaryFile("w", delete=False) as f:
            f.write(self.python_script)
            f.flush()
            self.copy_file_to_remote(local_path=f.name, remote_path=self.python_script_loc)

    def create_folder_on_remote(self, remote_path):
        remote_path = Path(remote_path)
        # using paramiko self.ssh, create the folder on the remote server
        if self.ssh is None:
            raise ConnectionError("SSH connection is not established")
        try:
            logger.info(f"Creating remote directory: {remote_path.as_posix()}")

            # Execute the mkdir command to create the folder on the remote server, and make it read/writable for the dattalab group
            stdin, stdout, stderr = self.ssh.exec_command(f"mkdir -p {remote_path.as_posix()} && chmod 2775 {remote_path.as_posix()}")

            # Check if there was any error
            error_message = stderr.read().decode().strip()
            if error_message:
                raise Exception(f"Error creating remote directory: {error_message}")

            logger.info(f"Successfully created remote directory: {remote_path.as_posix()}")
        except Exception as e:
            logger.error(f"Exception during directory creation: {str(e)}")
            raise

    def copy_file_to_remote(self, local_path, remote_path):
        local_path = Path(local_path)
        remote_path = Path(remote_path)
        if self.ssh is None:
            raise ConnectionError("SSH connection is not established")

        try:
            # Create an SFTP session from the SSH connection
            sftp = self.ssh.open_sftp()

            logger.info(f"Transferring {local_path} to {self.ssh_landing_hostname}:{remote_path.as_posix()}")
            sftp.put(local_path.as_posix(), remote_path.as_posix())

            logger.info(
                f"Successfully transferred {local_path.as_posix()} to {remote_path.as_posix()}"
            )

            # Close the SFTP session
            sftp.close()
        except Exception as e:
            logger.error(f"Exception during file transfer: {str(e)}")
            raise

    def copy_file_from_remote(self, remote_path, local_path):
        local_path = Path(local_path)
        remote_path = Path(remote_path)
        if self.ssh is None:
            raise ConnectionError("SSH connection is not established")

        try:
            # Create an SFTP session from the SSH connection
            sftp = self.ssh.open_sftp()

            logger.info(f"Transferring {self.ssh_landing_hostname}:{remote_path.as_posix()} to {local_path}")
            sftp.get(remote_path.as_posix(), local_path)

            logger.info(
                f"Successfully transferred {self.ssh_landing_hostname}:{remote_path.as_posix()} to {local_path}"
            )
            
            # Close the SFTP session
            sftp.close()
        except Exception as e:
            logger.error(f"Exception during file transfer: {str(e)}")
            raise

    def get_running_jobs_info_from_o2(self):
        # Run the squeue command to find the airflow_ssh_landing job
        login_ssh = paramiko.SSHClient()  # Connect to the login node
        login_ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        login_ssh.connect(self.o2_login_server, username=self.o2_username)
        squeue_cmd = "squeue --me -o '%.18i %.9P %.50j %.8u %.2t %.10M %.9l %.6D %R'"
        stdin, stdout, stderr = login_ssh.exec_command(squeue_cmd)
        slurm_output = stdout.read().decode()
        login_ssh.close() # Close the SSH connection to the login node
        running_jobs = parse_squeue_output(slurm_output)
        return running_jobs

    def find_ssh_landing_hostname(self):
        """ Find the hostname of the node where the airflow_ssh_landing job is running.
            If there is none, use the login node at your own risk.
        """
        running_jobs = self.get_running_jobs_info_from_o2()
        airflow_job = [job for job in running_jobs.values() if job['NAME'] == 'airflow_ssh_landing' and job['ST']=="R"]  # state == running
        if len(airflow_job) == 0:
            logging.warn("No airflow_ssh_landing job found, please start one: 'sbatch -p long -t 10-0 --mem 4GB -c 1 -J \"airflow_ssh_landing\" /home/jop9552/idle_forever.sh'")
            logging.warn("Using login node for now, but this is not good practice and may fail for many simultaneously running jobs.")
            self.ssh_landing_hostname = self.o2_login_server
        elif len(airflow_job) >= 1:
            airflow_job = airflow_job[0]
            self.ssh_landing_hostname = airflow_job['NODELIST(REASON)'] + ".o2.rc.hms.harvard.edu"
            logger.info("Found SSH landing job with hostname: " + self.ssh_landing_hostname)
        

    def establish_ssh_connection(self, n_attempts=5, attempt_delay=60):
        """Establish a sustained connection to the discovered SSH landing node using paramiko.
        """
        if self.ssh_landing_hostname is None:
            self.find_ssh_landing_hostname()
        self.ssh = paramiko.SSHClient()
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        connected = False
        ii = 0
        while not connected and ii < n_attempts:
            try:
                logger.info(f"Connecting to O2: {self.ssh_landing_hostname}")
                self.ssh.connect(self.ssh_landing_hostname, username=self.o2_username)
                connected = True
                logger.info("SSH connection established.")
            except Exception as e:
                logger.error(f"Error connecting to O2: {str(e)}")
                ii += 1
                logger.info(f"Retrying connection in {attempt_delay} seconds")
                time.sleep(attempt_delay)
        if not connected:
            raise ConnectionError("Could not establish SSH connection to O2") 

    def close_ssh_connection(self):
        self.ssh.close()
        self.ssh = None
        logger.info("SSH connection closed.")

    def submit(self):
        expected_slurm_output = "Submitted batch job "
        submit_command = f"sbatch {self.slurm_script_loc}"

        if self.ssh is None:
            raise ConnectionError("SSH connection is not established")

        stdin, stdout, stderr = self.ssh.exec_command(submit_command)
        slurm_output = stdout.read().decode()
        # check if the job was submitted successfully
        if slurm_output[: len(expected_slurm_output)] != expected_slurm_output:
            raise Exception(f"Error submitting job: {slurm_output}")
        # grab job id
        self.slurm_job_id = slurm_output[len(expected_slurm_output) : -1]
        logger.info(f"Job submitted successfully with job id: {self.slurm_job_id}")
        # self.close_ssh_connection()

    def cancel(self):
        if self.ssh is None:
            raise ConnectionError("SSH connection is not established")
        if not self.slurm_job_id:
            logger.warning("No job id set, cannot cancel job")
            return
        cancel_command = f"scancel {self.slurm_job_id}"
        stdin, stdout, stderr = self.ssh.exec_command(cancel_command)
        logger.info(f"Job cancelled: {self.slurm_job_id}")

    def check_job_success(self):
        # job success is specific to the job type
        raise NotImplementedError

    def check_job_status(self):

        if self.ssh is None:
            self.establish_ssh_connection(n_attempts=10, attempt_delay=np.random.randint(60, 300))
        
        # If still couldn't connect, try looking for a new landing hostname.
        if self.ssh is None:
            self.find_ssh_landing_hostname()
            self.establish_ssh_connection(n_attempts=10, attempt_delay=np.random.randint(60, 300))

        # If still couldn't connect, raise an error. It's annoying to fail this way b/c it leaves a job orphaned on O2.
        # But the alternative is just futily checking job status forever, which would be annoying.
        if self.ssh is None:
            raise ConnectionError("Could not establish SSH connection to O2")
            

        if not self.slurm_job_id:
            raise ValueError("slurm_job_id is not set")

        logger.info(f"Checking job status: {self.slurm_job_id}")
        # check_command = f"sacct -j {self.slurm_job_id} --format=State --noheader"
        check_command = f"sacct -j {self.slurm_job_id} --format=JobID,State | grep -E '^[0-9]+ ' | awk '{{print $2}}'"
        stdin, stdout, stderr = self.ssh.exec_command(check_command)
        slurm_output = stdout.read().decode()[:-1]
        job_state = slurm_output

        # Add custom handling based on the job state
        ret = False
        if job_state == "COMPLETED":
            logger.info("The job has finished successfully.")
            ret = True
        elif job_state == "PENDING":
            logger.info("The job is waiting to be scheduled.")
            
        elif job_state == "RUNNING":
            logger.info("The job is currently running.")
            
        elif job_state == "FAILED":
            logger.info("The job failed.")
            self.report_output_log()
            raise Exception("Job failed.")

        elif job_state == "CANCELLED":
            logger.info("The job was cancelled.")
            self.report_output_log()
            raise Exception("Job failed.")

        elif job_state == "CANCELLED+":
            logger.info("The job was cancelled.")
            self.report_output_log()
            raise Exception("Job failed.")

        elif job_state == "TIMEOUT":
            logger.info("The job has timed out.")
            self.report_output_log()
            raise Exception("Job failed.")

        elif job_state == "NODE_FAIL":
            logger.info("The job terminated due to node failure.")
            self.report_output_log()
            raise Exception("Job failed.")

        elif job_state == "OUT_OF_MEMORY":
            logger.info("The job was terminated due to exceeding memory limits.")

            self.report_output_log()
            raise Exception("Job failed.")

        elif job_state == "COMPLETING":
            logger.info("The job is in the process of completing.")
            
        elif job_state == "REQUEUED":
            logger.info("The job was requeued.")
            
        elif job_state == "RESIZING":
            logger.info("The job is being resized.")
            
        elif job_state == "SUSPENDED":
            logger.info("The job is suspended.")
            self.report_output_log()
            raise Exception("Job failed.")

        elif job_state == "SPECIAL_EXIT":
            logger.info("The job terminated with a special exit state.")
            self.report_output_log()
            raise Exception("Job failed.")

        elif "OUT_OF_ME" in job_state:
            logger.info("The job was terminated due to exceeding memory limits.")
            self.report_output_log()
            raise Exception("Job failed.")

        else:
            logger.info(f"Unknown job state: {job_state}")

        # Point: Close the SSH connection to free up resources
        # Counterpoint: O2 ssh connection is fairly finicky, actually, so better to just connect once and stay connected.
        # self.close_ssh_connection()

        return ret

def create_folder_on_remote(
    remote_path, username="tis697", remote_server="login.o2.rc.hms.harvard.edu"
):
    command = [
        "ssh",
        "-T",
        "-o",
        "BatchMode=yes",
        f"{username}@{remote_server}",
        f"mkdir -p {remote_path}",
    ]
    logger.info("Creating folder: " + " ".join(command))
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(f"Error creating folder: {result.stderr}")


def scp_file_to_remote(
    local_path, remote_path, username="tis697", remote_server="login.o2.rc.hms.harvard.edu"
):
    # Configure SCP command with optimized options
    command = [
        "scp",
        "-T",  # Disable pseudo-terminal allocation
        "-o",
        "Compression=no",  # Disable compression
        "-o",
        "IPQoS=throughput",  # Optimize for throughput
        local_path,
        f"{username}@{remote_server}:{remote_path}",
    ]

    # Execute the SCP command
    logger.info("Sending file to remote: " + " ".join(command))
    result = subprocess.run(command, capture_output=True, text=True)

    # Check for errors in the SCP operation and raise an exception if any
    if result.returncode != 0:
        raise Exception(f"Error transferring file to remote: {result.stderr}")
