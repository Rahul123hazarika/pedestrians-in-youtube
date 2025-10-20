
# my "main.py" file is for downloading videos from ftp and detections using yolov11, if detection of that file not working then use this code to detect as per video ids. 


from helper_script import Youtube_Helper
import os

# === Configuration ===
video_id = "lmLWoDdQgtQ"  # change this to your desired video ID
video_path = f"./videos/lmLWoDdQgtQ.mp4"
output_folder = "./outputs"

# === Initialize Helper ===
helper = Youtube_Helper()

if not os.path.exists(video_path):
    print(f" Video file not found: {video_path}")
    print("Please download it first using main.py or FTP.")
else:
    print(f"Using existing video: {video_path}")
    fps_val = helper.get_video_fps(video_path) or 30

    print(" Running object detection/tracking...")
    helper.tracking_mode(
        input_video_path=video_path,
        output_video_path=output_folder,
        video_title=video_id,
        video_fps=int(fps_val),
        seg_mode=False,      # disable segmentation
        bbox_mode=True,      # enable detection/tracking
        flag=1               # save annotated video
    )

    print(f"Detection complete for {video_id}")
    print(f"Output saved in: {output_folder}")
 
