"""
Data loading and preprocessing utilities for the Flickd Fashion Analysis Pipeline
"""
import json
import lzma
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import requests
from PIL import Image
import io
import logging
from tqdm import tqdm

from ..config import (
    IMAGES_CSV, PRODUCT_DATA_XLSX, VIBES_JSON, VIDEOS_DIR,
    PROCESSED_DIR, IMAGE_SIZE
)

logger = logging.getLogger(__name__)


class DataLoader:
    """Handles loading and preprocessing of all data sources"""
    
    def __init__(self):
        self.product_data = None
        self.images_data = None
        self.vibes_list = None
        self.video_metadata = {}
        
    def load_product_data(self) -> pd.DataFrame:
        """Load product data from Excel file"""
        try:
            if PRODUCT_DATA_XLSX.exists():
                self.product_data = pd.read_excel(PRODUCT_DATA_XLSX)
                logger.info(f"Loaded product data: {len(self.product_data)} products")
            else:
                logger.warning("Product data file not found, creating empty DataFrame")
                self.product_data = pd.DataFrame()
            return self.product_data
        except Exception as e:
            logger.error(f"Error loading product data: {e}")
            return pd.DataFrame()
    
    def load_images_data(self) -> pd.DataFrame:
        """Load images data from CSV file"""
        try:
            self.images_data = pd.read_csv(IMAGES_CSV)
            logger.info(f"Loaded images data: {len(self.images_data)} image entries")
            return self.images_data
        except Exception as e:
            logger.error(f"Error loading images data: {e}")
            return pd.DataFrame()
    
    def load_vibes_list(self) -> List[str]:
        """Load vibes list from JSON file"""
        try:
            with open(VIBES_JSON, 'r') as f:
                self.vibes_list = json.load(f)
            logger.info(f"Loaded vibes: {self.vibes_list}")
            return self.vibes_list
        except Exception as e:
            logger.error(f"Error loading vibes list: {e}")
            return []
    
    def load_video_metadata(self, video_id: str) -> Dict:
        """Load metadata for a specific video"""
        try:
            # Load compressed JSON metadata
            json_file = VIDEOS_DIR / f"{video_id}.json.xz"
            txt_file = VIDEOS_DIR / f"{video_id}.txt"
            
            metadata = {}
            
            # Load compressed JSON if exists
            if json_file.exists():
                with lzma.open(json_file, 'rt', encoding='utf-8') as f:
                    metadata['instagram_data'] = json.load(f)
            
            # Load text caption if exists
            if txt_file.exists():
                with open(txt_file, 'r', encoding='utf-8') as f:
                    metadata['caption'] = f.read().strip()
            
            self.video_metadata[video_id] = metadata
            return metadata
            
        except Exception as e:
            logger.error(f"Error loading metadata for {video_id}: {e}")
            return {}
    
    def get_video_list(self) -> List[str]:
        """Get list of available video IDs"""
        video_files = list(VIDEOS_DIR.glob("*.mp4"))
        video_ids = [f.stem for f in video_files]
        logger.info(f"Found {len(video_ids)} videos: {video_ids}")
        return video_ids
    
    def download_product_image(self, image_url: str, product_id: str) -> Optional[Image.Image]:
        """Download and process a product image"""
        try:
            # Check if image is already cached
            cache_path = PROCESSED_DIR / "images" / f"{product_id}.jpg"
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            
            if cache_path.exists():
                return Image.open(cache_path).convert('RGB')
            
            # Skip known broken URLs
            broken_patterns = ['DSCF1654', 'WEB12', 'DSCF1656']
            if any(pattern in image_url for pattern in broken_patterns):
                logger.warning(f"Skipping known broken URL: {image_url}")
                return None

            # Download image
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            
            # Process image
            image = Image.open(io.BytesIO(response.content)).convert('RGB')
            image = image.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
            
            # Cache image
            image.save(cache_path, 'JPEG', quality=85)
            
            return image
            
        except Exception as e:
            logger.error(f"Error downloading image {image_url}: {e}")
            return None
    
    def cache_all_product_images(self, limit: Optional[int] = None) -> Dict[str, str]:
        """Download and cache all product images"""
        if self.images_data is None:
            self.load_images_data()
        
        cached_images = {}
        
        # Get unique product images (take first image per product ID)
        unique_images = self.images_data.drop_duplicates(subset=['id']).head(limit) if limit else self.images_data.drop_duplicates(subset=['id'])
        
        logger.info(f"Caching {len(unique_images)} product images...")
        
        for _, row in tqdm(unique_images.iterrows(), total=len(unique_images), desc="Downloading images"):
            product_id = str(row['id'])
            image_url = row['image_url']
            
            cache_path = PROCESSED_DIR / "images" / f"{product_id}.jpg"
            
            if cache_path.exists():
                cached_images[product_id] = str(cache_path)
                continue
            
            image = self.download_product_image(image_url, product_id)
            if image:
                cached_images[product_id] = str(cache_path)
        
        logger.info(f"Successfully cached {len(cached_images)} images")
        return cached_images
