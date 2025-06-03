"""
Video processing and frame extraction for fashion item detection
"""
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
import logging
from PIL import Image

from ..config import VIDEOS_DIR, FRAME_EXTRACTION_INTERVAL, MAX_FRAMES_PER_VIDEO, IMAGE_SIZE

logger = logging.getLogger(__name__)


class VideoProcessor:
    """Handles video processing and frame extraction"""
    
    def __init__(self):
        self.current_video = None
        self.frames = []
    
    def extract_frames(self, video_id: str) -> List[np.ndarray]:
        """Extract frames from video at specified intervals"""
        video_path = VIDEOS_DIR / f"{video_id}.mp4"
        
        if not video_path.exists():
            logger.error(f"Video file not found: {video_path}")
            return []
        
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_interval = int(fps * FRAME_EXTRACTION_INTERVAL)
            
            frame_count = 0
            extracted_count = 0
            
            logger.info(f"Extracting frames from {video_id} (FPS: {fps})")
            
            while cap.isOpened() and extracted_count < MAX_FRAMES_PER_VIDEO:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % frame_interval == 0:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame_rgb)
                    extracted_count += 1
                
                frame_count += 1
            
            logger.info(f"Extracted {len(frames)} frames from {video_id}")
            
        except Exception as e:
            logger.error(f"Error extracting frames from {video_id}: {e}")
        finally:
            cap.release()
        
        self.frames = frames
        return frames
    
    def detect_scene_changes(self, video_id: str, threshold: float = 0.3) -> List[int]:
        """Detect scene changes in video for better frame selection"""
        video_path = VIDEOS_DIR / f"{video_id}.mp4"
        
        if not video_path.exists():
            return []
        
        cap = cv2.VideoCapture(str(video_path))
        scene_frames = []
        
        try:
            prev_frame = None
            frame_idx = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert to grayscale for comparison
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                if prev_frame is not None:
                    # Calculate histogram difference
                    hist1 = cv2.calcHist([prev_frame], [0], None, [256], [0, 256])
                    hist2 = cv2.calcHist([gray], [0], None, [256], [0, 256])
                    
                    # Compare histograms
                    correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
                    
                    if correlation < (1 - threshold):
                        scene_frames.append(frame_idx)
                
                prev_frame = gray
                frame_idx += 1
            
            logger.info(f"Detected {len(scene_frames)} scene changes in {video_id}")
            
        except Exception as e:
            logger.error(f"Error detecting scene changes in {video_id}: {e}")
        finally:
            cap.release()
        
        return scene_frames
    
    def extract_frames_at_scenes(self, video_id: str) -> List[np.ndarray]:
        """Extract frames at scene changes for better diversity"""
        scene_frames = self.detect_scene_changes(video_id)

        if not scene_frames:
            # Fallback to regular interval extraction
            return self.extract_frames(video_id)

        video_path = VIDEOS_DIR / f"{video_id}.mp4"
        cap = cv2.VideoCapture(str(video_path))
        frames = []

        try:
            # Add first frame
            scene_frames = [0] + scene_frames

            # Limit to MAX_FRAMES_PER_VIDEO
            scene_frames = scene_frames[:MAX_FRAMES_PER_VIDEO]

            for frame_idx in scene_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()

                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame_rgb)

            logger.info(f"Extracted {len(frames)} frames at scene changes from {video_id}")

        except Exception as e:
            logger.error(f"Error extracting frames at scenes from {video_id}: {e}")
        finally:
            cap.release()

        self.frames = frames
        return frames

    def extract_frames_from_path(self, video_path: str) -> List[np.ndarray]:
        """Extract frames from a custom video path (for new videos)"""
        video_path = Path(video_path)

        if not video_path.exists():
            logger.error(f"Video file not found: {video_path}")
            return []

        cap = cv2.VideoCapture(str(video_path))
        frames = []

        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_interval = int(fps * FRAME_EXTRACTION_INTERVAL)

            frame_count = 0
            extracted_count = 0

            logger.info(f"Extracting frames from {video_path.name} (FPS: {fps})")

            while cap.isOpened() and extracted_count < MAX_FRAMES_PER_VIDEO:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % frame_interval == 0:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame_rgb)
                    extracted_count += 1

                frame_count += 1

            logger.info(f"Extracted {len(frames)} frames from {video_path.name}")

        except Exception as e:
            logger.error(f"Error extracting frames from {video_path}: {e}")
        finally:
            cap.release()

        self.frames = frames
        return frames
    
    def preprocess_frame(self, frame: np.ndarray) -> Image.Image:
        """Preprocess frame for model input"""
        # Convert numpy array to PIL Image
        if isinstance(frame, np.ndarray):
            image = Image.fromarray(frame)
        else:
            image = frame
        
        # Resize to standard size
        image = image.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
        
        return image
    
    def get_video_info(self, video_id: str) -> Dict:
        """Get basic video information"""
        video_path = VIDEOS_DIR / f"{video_id}.mp4"
        
        if not video_path.exists():
            return {}
        
        cap = cv2.VideoCapture(str(video_path))
        
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            
            info = {
                "fps": fps,
                "frame_count": frame_count,
                "width": width,
                "height": height,
                "duration": duration
            }
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting video info for {video_id}: {e}")
            return {}
        finally:
            cap.release()
