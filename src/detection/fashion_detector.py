"""
Fashion item detection using YOLOv8 and custom fashion detection
"""
import torch
import numpy as np
from ultralytics import YOLO
from PIL import Image
from typing import List, Dict, Tuple, Optional
import logging
import cv2

from ..config import YOLO_MODEL, FASHION_ITEMS, MODELS_DIR, IMAGE_SIZE

logger = logging.getLogger(__name__)


class FashionDetector:
    """Detects fashion items in images using YOLOv8 and custom logic"""
    
    def __init__(self):
        self.yolo_model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_models()
    
    def load_models(self):
        """Load YOLO model for person detection"""
        try:
            model_path = MODELS_DIR / YOLO_MODEL
            self.yolo_model = YOLO(YOLO_MODEL)  # This will download if not exists
            logger.info(f"Loaded YOLO model on {self.device}")
        except Exception as e:
            logger.error(f"Error loading YOLO model: {e}")
    
    def detect_persons(self, image) -> List[Dict]:
        """Detect persons in image using YOLO"""
        if self.yolo_model is None:
            return []

        try:
            # Convert to numpy array for YOLO if needed
            if isinstance(image, Image.Image):
                img_array = np.array(image)
            else:
                img_array = image
            
            # Run YOLO detection with very low confidence and multiple scales for better person detection
            results = self.yolo_model(img_array, conf=0.05, iou=0.4, verbose=False, imgsz=640)
            
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Check if detection is a person (class 0 in COCO)
                        if int(box.cls) == 0:  # Person class
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            confidence = float(box.conf)
                            
                            detections.append({
                                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                                "confidence": confidence,
                                "class": "person"
                            })
            
            if detections:
                logger.info(f"Successfully detected {len(detections)} persons with confidences: {[d['confidence'] for d in detections]}")

            return detections
            
        except Exception as e:
            logger.error(f"Error in person detection: {e}")
            return []
    
    def crop_person_regions(self, image, detections: List[Dict]) -> List[Image.Image]:
        """Crop person regions from image"""
        crops = []

        # Convert to PIL Image if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        for detection in detections:
            try:
                bbox = detection["bbox"]
                x1, y1, x2, y2 = bbox

                # Add some padding
                padding = 20
                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(image.width, x2 + padding)
                y2 = min(image.height, y2 + padding)

                # Crop the region
                crop = image.crop((x1, y1, x2, y2))
                crops.append(crop)
                
            except Exception as e:
                logger.error(f"Error cropping person region: {e}")
                continue
        
        logger.info(f"Successfully cropped {len(crops)} person regions")
        return crops
    
    def analyze_fashion_items(self, person_crop: Image.Image, frame_number: int = 0) -> List[Dict]:
        """Analyze fashion items in a person crop using color and region analysis"""
        try:
            # Convert to numpy array
            img_array = np.array(person_crop)
            height, width = img_array.shape[:2]

            fashion_items = []

            # Define regions for different clothing types
            regions = {
                "top": (0, 0, width, height // 2),  # Upper half
                "bottom": (0, height // 2, width, height),  # Lower half
                "full": (0, 0, width, height)  # Full body for dresses
            }

            # Analyze each region
            for region_name, (x1, y1, x2, y2) in regions.items():
                region_crop = img_array[y1:y2, x1:x2]

                if region_crop.size == 0:
                    continue

                # Analyze colors in the region
                colors = self.extract_dominant_colors(region_crop)

                # Determine clothing type based on region and characteristics
                clothing_type = self.classify_clothing_type(region_name, region_crop, colors)

                if clothing_type:
                    # Convert bbox to (x, y, w, h) format as required
                    w, h = x2 - x1, y2 - y1

                    fashion_items.append({
                        "class_name": clothing_type,
                        "bbox": [x1, y1, w, h],  # (x, y, w, h) format as required
                        "confidence": 0.7,  # Default confidence for rule-based detection
                        "frame_number": frame_number,
                        "type": clothing_type,  # For compatibility with matching
                        "color": colors[0] if colors else "unknown",
                        "region": region_name
                    })

            return fashion_items

        except Exception as e:
            logger.error(f"Error analyzing fashion items: {e}")
            return []
    
    def extract_dominant_colors(self, image_region: np.ndarray, k: int = 3) -> List[str]:
        """Extract dominant colors from image region"""
        try:
            # Check if image region is valid
            if image_region is None or image_region.size == 0:
                return ["unknown"]

            # Ensure image has the right shape
            if len(image_region.shape) != 3 or image_region.shape[2] != 3:
                return ["unknown"]

            # Check minimum size
            if image_region.shape[0] < 5 or image_region.shape[1] < 5:
                return ["unknown"]

            # Reshape image to be a list of pixels
            pixels = image_region.reshape(-1, 3)

            # Remove any invalid pixels (NaN, inf, or out of range)
            valid_mask = (
                ~np.isnan(pixels).any(axis=1) &
                ~np.isinf(pixels).any(axis=1) &
                (pixels >= 0).all(axis=1) &
                (pixels <= 255).all(axis=1)
            )
            pixels = pixels[valid_mask]

            if len(pixels) < 10:  # Need at least 10 valid pixels
                return ["unknown"]

            # Limit number of pixels for performance
            if len(pixels) > 1000:
                indices = np.random.choice(len(pixels), 1000, replace=False)
                pixels = pixels[indices]

            # Use k-means clustering to find dominant colors
            from sklearn.cluster import KMeans

            n_clusters = min(k, len(pixels), 5)  # Max 5 clusters
            if n_clusters < 1:
                return ["unknown"]

            kmeans = KMeans(
                n_clusters=n_clusters,
                random_state=42,
                n_init=3,  # Reduced for speed
                max_iter=50  # Reduced for speed
            )
            kmeans.fit(pixels)

            # Get the colors
            colors = kmeans.cluster_centers_.astype(int)

            # Convert RGB to color names (simplified)
            color_names = []
            for color in colors:
                try:
                    color_name = self.rgb_to_color_name(color)
                    if color_name and color_name != "unknown":
                        color_names.append(color_name)
                except:
                    continue

            # Return at least one color
            if not color_names:
                return ["unknown"]

            return color_names[:3]  # Return top 3 colors

        except Exception as e:
            logger.error(f"Error extracting colors: {e}")
            return ["unknown"]
    
    def rgb_to_color_name(self, rgb: np.ndarray) -> str:
        """Convert RGB values to color name (simplified)"""
        r, g, b = rgb
        
        # Simple color classification
        if r > 200 and g > 200 and b > 200:
            return "white"
        elif r < 50 and g < 50 and b < 50:
            return "black"
        elif r > g and r > b:
            if r > 150:
                return "red"
            else:
                return "dark_red"
        elif g > r and g > b:
            return "green"
        elif b > r and b > g:
            return "blue"
        elif r > 150 and g > 150 and b < 100:
            return "yellow"
        elif r > 150 and g < 100 and b > 150:
            return "purple"
        elif r > 100 and g > 100 and b > 100:
            return "gray"
        else:
            return "mixed"
    
    def classify_clothing_type(self, region: str, image_region: np.ndarray, colors: List[str]) -> Optional[str]:
        """Classify clothing type based on region and characteristics"""
        height, width = image_region.shape[:2]
        aspect_ratio = height / width if width > 0 else 1
        
        if region == "top":
            # Upper body clothing
            if aspect_ratio > 1.5:
                return "dress"  # Might be a long dress
            else:
                return "top"
        elif region == "bottom":
            # Lower body clothing
            if aspect_ratio > 1.2:
                return "pants"
            else:
                return "shorts"
        elif region == "full":
            # Full body - check for dress
            if aspect_ratio > 2.0:
                return "dress"
        
        return None
    
    def map_coco_to_fashion(self, class_id: int) -> Optional[str]:
        """Map COCO class IDs to fashion item types"""
        # COCO class mappings to fashion items
        coco_to_fashion = {
            0: None,  # person - we'll handle separately
            25: "backpack",  # backpack
            26: "handbag",   # handbag
            27: "bag",       # tie -> bag (approximate)
            28: "bag",       # suitcase -> bag
            # We'll detect clothing through person analysis
        }
        return coco_to_fashion.get(class_id)

    def detect_fashion_items(self, image, frame_number: int = 0) -> List[Dict]:
        """Main method to detect fashion items in an image"""
        try:
            # Convert numpy array to PIL Image if needed
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)

            img_array = np.array(image)

            # Run YOLO detection for all objects
            results = self.yolo_model(img_array, conf=0.3, iou=0.4, verbose=False, imgsz=640)

            all_fashion_items = []

            # First, detect accessories and bags directly from YOLO
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        class_id = int(box.cls)
                        confidence = float(box.conf)
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                        # Convert to (x, y, w, h) format as required
                        x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)

                        # Map COCO classes to fashion items
                        fashion_class = self.map_coco_to_fashion(class_id)

                        if fashion_class:
                            # Extract color from the detected region
                            if y2 > y1 and x2 > x1:
                                region_crop = img_array[int(y1):int(y2), int(x1):int(x2)]
                                colors = self.extract_dominant_colors(region_crop)
                            else:
                                colors = ["unknown"]

                            fashion_item = {
                                "class_name": fashion_class,
                                "bbox": [x, y, w, h],  # (x, y, w, h) format as required
                                "confidence": round(confidence, 3),
                                "frame_number": frame_number,
                                "type": fashion_class,  # For compatibility with matching
                                "color": colors[0] if colors else "unknown"
                            }

                            all_fashion_items.append(fashion_item)

            # Now detect persons and analyze their clothing
            person_detections = self.detect_persons(image)

            if not person_detections:
                logger.warning("No persons detected in image with standard detection")
                # Try fallback detection with very low confidence
                try:
                    fallback_results = self.yolo_model(img_array, conf=0.01, iou=0.3, verbose=False, imgsz=320)
                    fallback_detections = []

                    for result in fallback_results:
                        boxes = result.boxes
                        if boxes is not None:
                            for i in range(len(boxes)):
                                class_id = int(boxes.cls[i])
                                if class_id == 0:  # person class
                                    confidence = float(boxes.conf[i])
                                    bbox = boxes.xyxy[i].cpu().numpy()
                                    fallback_detections.append({
                                        'bbox': bbox,
                                        'confidence': confidence,
                                        'class': 'person'
                                    })

                    if fallback_detections:
                        logger.info(f"Fallback detection found {len(fallback_detections)} persons")
                        person_detections = fallback_detections

                except Exception as e:
                    logger.error(f"Fallback detection failed: {e}")

            if person_detections:
                # Crop person regions
                person_crops = self.crop_person_regions(image, person_detections)

                # Analyze each person crop for clothing
                for i, crop in enumerate(person_crops):
                    fashion_items = self.analyze_fashion_items(crop, frame_number)

                    # Add person index to each item
                    for item in fashion_items:
                        item["person_id"] = i

                    all_fashion_items.extend(fashion_items)

            logger.info(f"Detected {len(all_fashion_items)} fashion items in frame {frame_number}")
            return all_fashion_items

        except Exception as e:
            logger.error(f"Error in fashion item detection: {e}")
            return []
