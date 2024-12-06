import logging
import multiprocessing
import os
import sys
from pathlib import Path
import av
import subprocess

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info(f"Python interpreter binary location: {sys.executable}")

# duration multiplier is about ~7 for 120 fps

def get_video_len(vid):
    c = av.open(Path(vid).as_posix())
    return c.streams.video[0].frames

class VideoCompressor:
    """
    Compresses videos in-place using ffmpeg. Relies on a bytes-per-frame threshold to determine whether to compress the video.

    See discussion here about different ffmpeg presets: https://superuser.com/questions/1556953/why-does-preset-veryfast-in-ffmpeg-generate-the-most-compressed-file-compared
    The takeaway is, if you use a "fast" preset, you might get what looks like the same file size — but the quality will be worse.
    Given that we're optimizing for data integrity and we have plenty of CPU time, the "slow" preset is a good choice.

    Parameters
    ----------
    recording_directory : str
        Directory where the videos are stored. Will compress videos in-place in this directory.

    output_directory_log : str
        Directory where the logs will be saved.

    preset : str
        Preset for compression. Options are: slow (TODO: add more options)

    post_comprn_max_kb_per_frame : int
        Maximum kilobytes per frame for a video to be considered compressed.

    crf : int
        Constant Rate Factor (CRF) for compression. Lower values mean better quality, but larger file sizes.

    nthreads : int
        Number of threads to use for compression. Default is -1, which will use all available threads.

    fmtgray : bool
        Convert the video to grayscale before compressing.

    force_framerate : float
        Force a specific framerate for the video. If None, will use the original framerate.

    patterns_to_exclude_from_vids : list
        List of patterns to exclude from the video list.

    recompute_completed : bool
        Whether to recompute videos that have already been compressed.

    Returns
    -------
    None

    """
    def __init__(
        self,
        recording_directory,
        output_directory_log,  # where to save the logs
        preset="slow",
        post_comprn_max_kb_per_frame=25,
        crf=21,
        nthreads=-1,
        fmtgray=False,
        force_framerate=None,
        patterns_to_exclude_from_vids=["azure", "TRIM", "tmp"],
        recompute_completed=False,
    ):
        self.recording_directory = Path(recording_directory)
        self.output_directory_log = Path(output_directory_log)
        self.preset = preset
        self.post_comprn_max_kb_per_frame = post_comprn_max_kb_per_frame
        self.crf = crf
        self.nthreads = nthreads
        self.fmtgray = fmtgray
        self.force_framerate = force_framerate
        self.patterns_to_exclude_from_vids = patterns_to_exclude_from_vids
        self.recompute_completed = recompute_completed        
    
    def check_completed(self):
        return (self.output_directory_log / "completed.txt").exists()

    def find_videos(self):
        self.all_videos = list(self.recording_directory.glob("*.mp4"))
        self.all_videos = [v for v in self.all_videos if not any([p in v.name for p in self.patterns_to_exclude_from_vids])]
        self.video_lengths = {v: get_video_len(v) for v in self.all_videos}
        logger.info(f"Processing {len(self.all_videos)} videos")
        assert len(self.all_videos) > 0, f"No videos found in {self.recording_directory}"

    def check_if_vid_compressed(self, vid):
        vid_size = os.path.getsize(vid)
        vid_nframes = self.video_lengths[vid]
        vid_kb_per_frame = vid_size / vid_nframes / 1e3
        if vid_kb_per_frame > self.post_comprn_max_kb_per_frame:
            return False
        else:
            return True

    def get_fps_for_ffmpeg(self, vid):
        if self.force_framerate is not None:
            original_framerate = float(self.force_framerate)
            assert isinstance(original_framerate, float), "Forced framerate must be a float."
        else:
            container = av.open(vid.as_posix())
            stream = container.streams.video[0]
            original_framerate = float(stream.average_rate)
            container.close()
            logger.info(f"Original framerate detected: {original_framerate} fps")
        return original_framerate

    def run(self):
        
        # Check for completion log file, skip if present unless overwriting
        if self.check_completed() and not self.recompute_completed:
            logger.info("Compression step already completed, skipping")
            return
        
        # Count num cores available
        if isinstance(self.nthreads, str):
            self.nthreads = int(self.nthreads)
        if self.nthreads == -1:    
            try:
                # assume we're on slurm
                self.nthreads = int(os.getenv('SLURM_CPUS_PER_TASK'))
            except KeyError:
                self.nthreads = int(multiprocessing.cpu_count())
        else:
            self.nthreads = int(self.nthreads)
        if self.nthreads is None or self.nthreads < 1:
            self.nthreads = 1


        # Find videos and get video lengths
        self.find_videos()

        for vid in self.all_videos:

            # Skip if they seem compressed already
            if self.check_if_vid_compressed(vid):
                logger.info(f"Video {vid} already compressed below threshold of {self.post_comprn_max_kb_per_frame}, skipping..")
                continue

            # Prepare to run compression
            logger.info(f"Running compression on {vid} with presest {self.preset}")
            fps = self.get_fps_for_ffmpeg(vid)

            # Count frames before compression, to ensure they match after compression
            num_frames = self.video_lengths[vid]
            original_filesize = os.path.getsize(vid)

            # Set up tmp output file
            replace_original = True  # Hardcode this for airflow pipeline
            output_vid = vid.as_posix().replace(".mp4", ".tmp.mp4")
            output_dir = os.path.dirname(output_vid)
            vid_name = os.path.basename(vid)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            if os.path.exists(output_vid):
                logger.error(f"Output video {output_vid} already exists — a previous job may have been interrupted. Fix manually.")
                raise ValueError()

            logger.info(f"Output video will be saved to: {output_vid}")
            if replace_original:
                logger.info("The original video will be replaced with the compressed video!")

            # Set ffmpeg options
            if self.fmtgray:
                fmt_filter = "-vf format=gray,format=yuv420p"
            else:
                fmt_filter = ""
            
            # Prep the ffmpeg command
            ffmpeg_command = f"module load ffmpeg; ffmpeg -y -r {fps} -i {vid} {fmt_filter} -c:v libx264 -preset {self.preset} -crf {self.crf} -threads {self.nthreads} {output_vid}"
            ffmpeg_command += " 2>&1"  # Capture stderr
            logger.info(f"Running ffmpeg command: {ffmpeg_command}")

            # Compress video, and read out the result
            # sys_out = os.popen(ffmpeg_command).read()
            # print(sys_out)
            # with open(f"{output_dir}/{vid_name}_COMPRESSION_log.txt", "w") as f:
            #     f.write(sys_out)  # Save the ffmpeg logs
            # with open(self.output_directory_log / f"{vid_name}_COMPRESSION_log.txt", "w") as f:
            #     f.write(sys_out)
            with (
                open(f"{output_dir}/{vid_name}_COMPRESSION_log.txt", "w") as f1,
                open(self.output_directory_log / f"{vid_name}_COMPRESSION_log.txt", "w") as f2,
            ):
                # Start the subprocess
                process = subprocess.Popen(
                    ffmpeg_command,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True  # Ensure the output is text, not bytes
                )
                
                # Iterate over the output line by line
                for line in iter(process.stdout.readline, ''):
                    print(line, end='')  # Print to console
                    f1.write(line)  # Write to file
                    f2.write(line)
                    f1.flush()  # Ensure the line is written immediately
                    f2.flush()
                process.stdout.close()
                process.wait()  # Wait for the process to complete

            # Count frames of output vid, ensure matching
            logger.info("Checking number of frames in new video")
            compressed_num_frames = get_video_len(output_vid)
            if num_frames != compressed_num_frames:
                raise ValueError(
                    f"Number of frames do not match, original vid has {num_frames} and compressed vid has {compressed_num_frames} frames."
                )
            
            # Check file size
            compressed_filesize = os.path.getsize(output_vid)
            logger.info(f"Original file size: {original_filesize/1e9:0.3f} GB")
            logger.info(f"Compressed file size: {compressed_filesize/1e9:0.3f} GB")
            logger.info(f"Compression ratio: {compressed_filesize/original_filesize:0.3f}")

            # Remove the original file if requested
            if (num_frames == compressed_num_frames):
                logger.info("Removing old video!")
                os.remove(vid)

            # Replace the original file with the compressed file if requested
            if replace_original:
                logger.info("Replacing original file with compressed file!")
                os.rename(output_vid, vid)

        # Write file to mark completion for airflow
        with open(self.output_directory_log / "completed.txt", "w") as f:
            f.write("completed")

        logger.info("Done.")
