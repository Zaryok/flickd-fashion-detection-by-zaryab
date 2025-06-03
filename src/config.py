"""
Configuration settings for the Flickd Fashion Analysis Pipeline
"""
import os
from pathlib import Path

# Base paths (Hackathon submission structure)
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
VIDEOS_DIR = DATA_DIR / "videos"  # Videos moved to data/videos
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = PROJECT_ROOT / "models"
FRAMES_DIR = PROJECT_ROOT / "frames"  # Required by hackathon format
API_DIR = PROJECT_ROOT / "api"  # Optional API endpoints

# Data files (now in data directory)
IMAGES_CSV = DATA_DIR / "images.csv"  # Product catalog
PRODUCT_DATA_XLSX = DATA_DIR / "product_data.xlsx"
VIBES_JSON = DATA_DIR / "vibes_list.json"  # Fashion vibes

# Model configurations
YOLO_MODEL = "yolov8n.pt"  # Nano version for faster processing
CLIP_MODEL = "openai/clip-vit-base-patch32"
NLP_MODEL = "distilbert-base-uncased"

# Fashion item classes for YOLO detection
FASHION_CLASSES = {
    0: "person",
    # We'll focus on person detection and then crop fashion items
    # Custom fashion detection will be implemented
}

# Fashion item types we want to detect
FASHION_ITEMS = [
    "dress", "top", "shirt", "blouse", "sweater", "jacket", "coat",
    "pants", "jeans", "skirt", "shorts", "shoes", "boots", "sneakers",
    "bag", "handbag", "backpack", "hat", "cap", "jewelry", "accessories"
]

# Similarity thresholds
SIMILARITY_THRESHOLDS = {
    "exact": 0.9,
    "similar": 0.75,
    "minimum": 0.75
}

# Processing parameters
FRAME_EXTRACTION_INTERVAL = 0.5  # seconds
MAX_FRAMES_PER_VIDEO = 15  # Reduced to prevent memory issues
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16  # Reduced batch size for stability
MAX_PROCESSING_TIME_PER_VIDEO = 300  # 5 minutes max per video
MAX_PROCESSING_TIME_PER_FRAME = 30  # 30 seconds max per frame

# Vibe classification keywords
VIBE_KEYWORDS = {
    "Coquette": [
        "feminine", "romantic", "soft", "delicate", "pink", "bow", "lace", 
        "floral", "sweet", "girly", "cute", "pastel", "vintage", "dreamy"
    ],
    "Clean Girl": [
        "minimal", "natural", "effortless", "simple", "clean", "fresh",
        "no-makeup", "dewy", "glowing", "understated", "casual", "basic"
    ],
    "Cottagecore": [
        "rustic", "countryside", "vintage", "floral", "nature", "garden",
        "cottage", "pastoral", "earthy", "handmade", "cozy", "rural"
    ],
    "Streetcore": [
        "urban", "street", "edgy", "cool", "trendy", "modern", "city",
        "casual", "sporty", "contemporary", "hip", "stylish"
    ],
    "Y2K": [
        "2000s", "metallic", "futuristic", "cyber", "tech", "digital",
        "holographic", "neon", "retro-future", "millennium", "space-age"
    ],
    "Boho": [
        "bohemian", "free-spirited", "ethnic", "tribal", "flowing", "loose",
        "hippie", "artistic", "eclectic", "wanderlust", "festival", "gypsy"
    ],
    "Party Glam": [
        "glamorous", "sparkly", "sequins", "glitter", "party", "night",
        "elegant", "sophisticated", "dressy", "formal", "luxurious", "chic"
    ]
}

# Create directories if they don't exist (Hackathon submission structure)
for directory in [DATA_DIR, PROCESSED_DIR, OUTPUTS_DIR, MODELS_DIR, FRAMES_DIR, API_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
