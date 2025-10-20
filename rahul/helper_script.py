# helper_script.py , interactive FTP + YOLO Video Analyzer


import os
import re
import time
import shutil
import datetime
import pathlib
import cv2
import torch
import pandas as pd
import numpy as np
import requests
import yaml
from tqdm import tqdm
from pytubefix import YouTube
from pytubefix.cli import on_progress
from moviepy.video.io.VideoFileClip import VideoFileClip
from ultralytics import YOLO
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from custom_logger import CustomLogger
import common

# ------------------ LOGGER ------------------
logger = CustomLogger(__name__)
os.makedirs("runs", exist_ok=True)

# ------------------ CONSTANTS ------------------
LINE_THICKNESS = 1
RENDER = False
SHOW_LABELS = False
SHOW_CONF = False


class Youtube_Helper:
    """Helper class for downloading, trimming, and detecting videos (FTP + YouTube)."""

    def __init__(self, video_title=None):
        self.tracking_model = common.get_configs("tracking_model")
        self.segment_model = common.get_configs("segment_model")
        self.bbox_tracker = common.get_configs("bbox_tracker")
        self.seg_tracker = common.get_configs("seg_tracker")
        self.mapping = pd.read_csv(common.get_configs("mapping"))
        self.confidence = common.get_configs("min_confidence")
        self.display_frame_tracking = common.get_configs("display_frame_tracking")
        self.display_frame_segmentation = common.get_configs("display_frame_segmentation")
        self.save_annoted_img = common.get_configs("save_annoted_img")
        self.save_tracked_img = common.get_configs("save_tracked_img")
        self.delete_labels = common.get_configs("delete_labels")
        self.delete_frames = common.get_configs("delete_frames")
        self.video_title = video_title or "video"
        self.client = common.get_configs("client")
        self.need_authentication = common.get_configs("need_authentication")
        self.update_package = common.get_configs("update_package")

    # ------------------ FTP VIDEO ACCESS ------------------
    def download_videos_from_ftp(self, filename, base_url, out_dir=".", username=None, password=None, timeout=20):
        """Download a video from FTP-like server using username/password authentication."""
        if not (base_url and username and password):
            logger.error("FTP credentials or base URL missing.")
            return None

        filename_mp4 = filename if filename.endswith(".mp4") else f"{filename}.mp4"
        os.makedirs(out_dir, exist_ok=True)

        try:
            with requests.Session() as session:
                session.auth = (username, password)
                session.headers.update({"User-Agent": "ftp-video-downloader/1.0"})

                # FTP has directory structure /v/tue1/browse etc.
                for alias in ["tue1", "tue2", "tue3"]:
                    browse_url = urljoin(base_url, f"v/{alias}/browse")
                    logger.info(f"Searching in: {browse_url}")

                    resp = session.get(browse_url, timeout=timeout)
                    if resp.status_code != 200:
                        continue

                    soup = BeautifulSoup(resp.text, "html.parser")
                    for a in soup.find_all("a"):
                        href = a.get("href", "")
                        if filename_mp4 in href:
                            file_url = urljoin(base_url, href)
                            local_path = os.path.join(out_dir, filename_mp4)

                            with session.get(file_url, stream=True) as r:
                                r.raise_for_status()
                                total = int(r.headers.get("content-length", 0))
                                with open(local_path, "wb") as f, tqdm(
                                    total=total, unit="B", unit_scale=True,
                                    desc=f"Downloading {filename_mp4}"
                                ) as bar:
                                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                                        if chunk:
                                            f.write(chunk)
                                            bar.update(len(chunk))
                            fps = self.get_video_fps(local_path)
                            res = self.get_video_resolution_label(local_path)
                            logger.info(f" Downloaded {filename_mp4} ({res}, {fps} FPS)")
                            return local_path, filename, res, fps
            logger.warning(f"{filename_mp4} not found on FTP server.")
            return None
        except Exception as e:
            logger.error(f"FTP download failed: {e}")
            return None

    # ------------------ YOUTUBE FALLBACK ------------------
    def download_video_with_resolution(self, vid, resolutions=["720p", "480p", "360p"], output_path="."):
        """Download YouTube video using pytubefix with preferred resolutions."""
        try:
            yt_url = f"https://www.youtube.com/watch?v={vid}"
            if self.need_authentication:
                yt = YouTube(yt_url, self.client, use_oauth=True, allow_oauth_cache=True, on_progress_callback=on_progress)
            else:
                yt = YouTube(yt_url, self.client, on_progress_callback=on_progress)

            for res in resolutions:
                stream = yt.streams.filter(res=res).first()
                if stream:
                    output_file = os.path.join(output_path, f"{vid}.mp4")
                    logger.info(f"Downloading {vid} in {res}...")
                    stream.download(output_path, filename=f"{vid}.mp4")
                    fps = self.get_video_fps(output_file)
                    logger.info(f" Download complete: {output_file}")
                    return output_file, vid, res, fps
            logger.error(f"No matching resolution found for {vid}")
            return None
        except Exception as e:
            logger.error(f"YouTube download failed: {e}")
            return None

    # ------------------ VIDEO UTILITIES ------------------
    def get_video_fps(self, video_file_path):
        """Return FPS of a video using OpenCV."""
        try:
            vid = cv2.VideoCapture(video_file_path)
            fps = vid.get(cv2.CAP_PROP_FPS)
            vid.release()
            return round(fps, 0)
        except Exception as e:
            logger.error(f"FPS retrieval failed: {e}")
            return 30.0

    @staticmethod
    def get_video_resolution_label(video_path):
        """Return resolution label (e.g., 720p)."""
        cap = cv2.VideoCapture(video_path)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        resolutions = {144: "144p", 240: "240p", 360: "360p", 480: "480p", 720: "720p", 1080: "1080p", 1440: "1440p"}
        return resolutions.get(height, f"{height}p")

    def trim_video(self, input_path, output_path, start_time, end_time):
        """Trim a video between start and end times."""
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            clip = VideoFileClip(input_path).subclip(start_time, end_time)
            clip.write_videofile(output_path, codec="libx264", audio_codec="aac", verbose=False, logger=None)
            clip.close()
            logger.info(f"Trimmed video saved: {output_path}")
        except Exception as e:
            logger.error(f"Trimming failed: {e}")

    # ------------------ YOLO TRACKING ------------------
    def update_track_buffer_in_yaml(self, yaml_path, video_fps):
        """Update YAML file with track buffer (based on fps)."""
        with open(yaml_path, "r") as f:
            cfg = yaml.safe_load(f)
        cfg["track_buffer"] = common.get_configs("track_buffer_sec") * video_fps
        with open(yaml_path, "w") as f:
            yaml.dump(cfg, f)

    def tracking_mode(self, input_video_path, output_video_path, video_title, video_fps,
                      seg_mode=False, bbox_mode=True, flag=0):
        """Run YOLOv11 object tracking and save annotated video."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if bbox_mode:
            self.update_track_buffer_in_yaml(self.bbox_tracker, video_fps)
            model = YOLO(self.tracking_model)
        elif seg_mode:
            self.update_track_buffer_in_yaml(self.seg_tracker, video_fps)
            model = YOLO(self.segment_model)
        else:
            return

        cap = cv2.VideoCapture(input_video_path)
        frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        out_path = os.path.join(output_video_path, f"{video_title}_detected.mp4")
        os.makedirs(output_video_path, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(out_path, fourcc, video_fps, (frame_width, frame_height))

        progress = tqdm(total=total_frames, desc="Detecting", unit="frames")

        while True:
            success, frame = cap.read()
            if not success:
                break
            try:
                results = model.track(frame, tracker=self.bbox_tracker if bbox_mode else self.seg_tracker,
                                      persist=True, conf=self.confidence, verbose=False, device=device)
                annotated = results[0].plot()
                out.write(annotated)

                if self.display_frame_tracking:
                    cv2.imshow("YOLOv11 Detection", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
            except Exception as e:
                logger.error(f"Frame error: {e}")
            progress.update(1)

        cap.release()
        out.release()
        progress.close()
        cv2.destroyAllWindows()
        logger.info(f" YOLO detection finished. Output saved to {out_path}")
