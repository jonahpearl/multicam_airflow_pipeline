import os
from pathlib import Path

def split_multicam_filename(filename, mode="recording"):
    """
    Splits a multicam filename into its components.

    Parameters
    ----------
    filename : str or Path
        The filename to split.

    mode: str
        If recording, behaves as described below, for parsing 6cam recording file names.
        If "validation_videos", expects an additional suffix in the filename, and parses accordingly.
            Eg "20240101_mouse1.top.with_2D_keypoints.mp4" or "20240101_mouse1.top.0.with_2D_keypoints.mp4"

    Returns
    -------
    filename_info : dict
        A dictionary containing the following
        - rec_name : str (ie 20240101_mouse1 or 24-01-01_12-00-00 or whatever you want)
        - camera : str (one of "top", "bottom", "side1", "side2", "side3", "side4""azure_top", "azure_top_depth", etc.)
        - start_frame : int (ie 0, 1000, etc.). Inferred as 0 if not present in filename.
        - ext : str (ie .mp4, .avi, etc.)
        
    Description:
    We have agreed on the following basic structure for the name of the recordings:
        recording_name = [recording base name].[camera info].[video info]

    Where:
        recording base name: the name of the recording, eg "test_recording". 
            This MUST NOT contain any dots.
            This may or may not include the date and time, depending on the append_datetime flag.
            Valid examples: "20240101_my_mouse", "test_recording", "test_recording_21-03-01-12-00-00-000000"

        camera info: the name of the camera, eg "top", "bottom", "side". This may also include the serial number of the camera.
            Valid examples: "top", "bottom", "side3", "top.12345678", "azure_top", "azure_top_depth"

        video info: the type of video, and, if max_video_frames is set in the config, the first frame number of the video.
            Valid examples: ".mp4", "0.mp4" [i.e. the first video from this Writer], "1000.mp4" [i.e. a video starting with the 1000th frame from this Writer]
        

    """
    filename = Path(filename).as_posix()
    basename = os.path.basename(filename)
    parts = basename.split(".")
    
    if mode == "recording":
        if len(parts) == 3:
            # rec_name.camera.ext
            rec_name = parts[0]
            camera = parts[1]
            ext = parts[2]
            start_frame = 0
        elif len(parts) == 4:
            # rec_name.camera.start_frame.ext
            rec_name = parts[0]
            camera = parts[1]
            start_frame = int(parts[2])
            ext = parts[3]
        elif len(parts) == 5:
            # rec_name.camera.start_frame.SUFFIX.ext  (used in keypoint validation videos)
            rec_name = parts[0]
            camera = parts[1]
            start_frame = int(parts[2])
            suffix = parts[3]
            ext = parts[4]
        else:
            raise ValueError(f"Invalid recording name: {rec_name}. Expected 3 or 4 '.'-separated parts, instead got {len(parts)}.")
        d = {
            "rec_name": rec_name,
            "camera": camera,
            "start_frame": start_frame,
            "ext": ext
        }
    elif mode == "validation_videos":
        if len(parts) == 4:
            # rec_name.camera.SUFFIX.ext
            rec_name = parts[0]
            camera = parts[1]
            suffix = parts[2]
            ext = parts[3]
            start_frame = 0
        elif len(parts) == 5:
            # rec_name.camera.start_frame.SUFFIX.ext
            rec_name = parts[0]
            camera = parts[1]
            start_frame = int(parts[2])
            suffix = parts[3]
            ext = parts[4]
        else:
            raise ValueError(f"Invalid recording name: {rec_name}. Expected 3 or 4 '.'-separated parts, instead got {len(parts)}.")
        d = {
            "rec_name": rec_name,
            "camera": camera,
            "start_frame": start_frame,
            "ext": ext,
            "suffix": suffix
        }
    else:
        raise ValueError(f"Invalid mode: {mode}. Expected 'recording' or 'validation_videos'.")

    return d