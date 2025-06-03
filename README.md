# 🎬 Flickd Fashion Analysis Pipeline

**AI-Powered Fashion Video Analysis - Hackathon Submission**

Complete implementation for the Flickd AI Hackathon featuring object detection, product matching, and vibe classification using YOLOv8, CLIP, FAISS, and DistilBERT.

## 🏆 Hackathon Requirements Compliance

✅ **Object Detection**: YOLOv8 for fashion item detection (tops, bottoms, dresses, jackets, accessories, bags, shoes)
✅ **Product Matching**: CLIP + FAISS for similarity search against catalog
✅ **Vibe Classification**: NLP-based classification into 7 fashion vibes
✅ **Output Format**: Exact JSON structure as specified
✅ **Similarity Thresholds**: Exact (>0.9), Similar (0.75-0.9), No Match (<0.75)
✅ **Tech Stack**: YOLOv8, CLIP, FAISS, spaCy/DistilBERT
✅ **Required Fields**: bbox (x,y,w,h), confidence scores, frame numbers

## 🚀 Complete Setup + Installation Instructions

### System Requirements
- **Python**: 3.8+ (3.9-3.11 recommended)
- **RAM**: 8GB+ recommended (4GB minimum)
- **Storage**: 5GB+ free space for models and cache
- **Internet**: Required for initial model downloads
- **OS**: Windows, macOS, or Linux

### Step-by-Step Installation

#### 1. Clone/Download the Repository
```bash
git clone <repository-url>
cd flickd-fashion-analysis
```

#### 2. Create Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

#### 3. Install Core Dependencies
```bash
# Upgrade pip first
pip install --upgrade pip

# Install all required packages
pip install -r requirements.txt
```

#### 4. Install Additional NLP Models (Optional but Recommended)
```bash
# Install spaCy English model for better NLP processing
python -m spacy download en_core_web_sm
```

#### 5. Verify Installation
```bash
# Quick test
python -c "import torch, transformers, ultralytics, cv2, faiss; print('All dependencies installed successfully!')"

# Comprehensive verification (recommended)
python verify_installation.py
```

### 🔧 Manual Dependency Installation (If requirements.txt fails)

If `pip install -r requirements.txt` fails, install packages individually:

```bash
# Core ML frameworks
pip install torch>=2.0.0 torchvision>=0.15.0
pip install transformers>=4.30.0
pip install ultralytics>=8.0.0

# Computer Vision
pip install opencv-python>=4.8.0
pip install Pillow>=10.0.0

# Data Processing
pip install pandas>=2.0.0 numpy>=1.24.0
pip install scikit-learn>=1.3.0

# Similarity Search
pip install faiss-cpu>=1.7.0

# Utilities
pip install requests>=2.31.0 tqdm>=4.65.0 pyyaml>=6.0
```

### 🚀 Running the System

#### Main Entry Point (Process All Videos)
```bash
python main.py
```

#### Alternative Commands
```bash
python main.py --all              # Process all videos explicitly
python main.py --video-id <ID>    # Process specific video
python main.py --stats            # Show pipeline statistics
```

### 🛠 Troubleshooting Common Issues

#### Issue: "No module named 'torch'"
```bash
# Solution: Install PyTorch
pip install torch torchvision
```

#### Issue: "CUDA out of memory"
```bash
# Solution: The system automatically uses CPU if GPU memory is insufficient
# No action needed - system will continue with CPU processing
```

#### Issue: "Failed to download model"
```bash
# Solution: Ensure internet connection and try again
# Models are downloaded automatically on first run
```

#### Issue: "Permission denied" on Windows
```bash
# Solution: Run command prompt as Administrator
# Or use: python -m pip install -r requirements.txt
```

### 📦 What Gets Downloaded Automatically

On first run, the system will automatically download:
- **YOLOv8 model** (~6MB) - for object detection
- **CLIP model** (~600MB) - for image embeddings
- **DistilBERT model** (~250MB) - for text classification
- **Product embeddings** - cached locally for faster processing

**Total download size**: ~1GB (one-time download)

### 📁 Required Data Structure

Ensure your project has this structure before running:
```
flickd-fashion-analysis/
├── data/
│   ├── videos/                    # Place your 6 MP4 video files here
│   ├── images.csv                 # Product catalog (provided)
│   ├── vibes_list.json           # Fashion vibes (provided)
│   └── processed/                 # Auto-created for cached data
├── outputs/                       # Auto-created for results
├── models/                        # Auto-created for cached models
├── frames/                        # Auto-created for frame extraction
└── api/                          # Auto-created for API endpoints
```

### 🔍 Environment Compatibility

**Tested Environments:**
- ✅ **Windows 10/11** with Python 3.9-3.11
- ✅ **macOS** (Intel & Apple Silicon) with Python 3.8-3.11
- ✅ **Linux Ubuntu 20.04+** with Python 3.8-3.11
- ✅ **Google Colab** (with GPU support)
- ✅ **Docker containers** (Python 3.9+ base images)

**GPU Support:**
- **NVIDIA GPU**: Automatically detected and used if available
- **CPU Only**: Fully supported (slower but works everywhere)
- **Apple M1/M2**: Supported via MPS backend

### 🚀 Quick Start (TL;DR)

For experienced users who want to get started immediately:
```bash
# 1. Install Python 3.8+ and pip
# 2. Clone repository and navigate to directory
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python main.py
```

### ✅ How to Know It's Working

When you run `python main.py`, you should see:

1. **Setup Phase** (~10 seconds):
   ```
   🚀 Flickd Fashion Analysis Pipeline Starting...
   📊 Loading data and initializing models...
   ✅ Setup completed in X.XX seconds
   ```

2. **Processing Phase** (~2-3 minutes total):
   ```
   🎬 Processing video 1/6: 2025-05-22_08-25-12_UTC
   📹 Extracting frames from video...
   🔍 Processing frame 1/15
   👗 Detected X fashion items in frame
   ✅ Completed 2025-05-22_08-25-12_UTC in XX.XX seconds
   ```

3. **Completion**:
   ```
   🎉 Pipeline completed successfully!
   📊 Final Summary:
   - Total Videos Processed: 6
   - Total Persons Detected: XX
   - Total Fashion Items: XXX
   📁 Results saved to outputs/ directory
   ```

4. **Output Files Created**:
   - `outputs/[video_id]_analysis.json` (6 files)
   - `outputs/final_results.json`

### 🎯 Expected Performance

- **Total Runtime**: 2-5 minutes (depending on hardware)
- **Memory Usage**: 2-4GB RAM
- **Storage**: ~1GB for models + ~100MB for cache
- **Success Rate**: Should process all 6 videos without errors

## 🎯 What This System Does

- **Video Processing**: Extracts key frames from fashion videos using scene detection
- **Object Detection**: Detects clothing items and accessories using YOLOv8
- **Product Matching**: Matches detected items to catalog using CLIP embeddings + FAISS similarity search
- **Vibe Classification**: Classifies fashion vibes using DistilBERT NLP models
- **Smart Caching**: Optimizes performance with intelligent embedding caching

## 📁 Project Structure (Hackathon Submission Format)

```
flickd-fashion-analysis/
├── /frames                        # Frame extraction (as required)
├── /models                        # Cached ML models and FAISS indices
├── /data                          # Hackathon dataset
│   ├── videos/                    # 6 MP4 videos with metadata
│   ├── images.csv                 # 11,691 product images with Shopify URLs
│   ├── processed/                 # Cached images and processed data
│   └── vibes_list.json           # 7 aesthetic categories
├── /api                          # API endpoints (optional)
├── src/                          # Source code modules
│   ├── preprocessing/            # Data loading and preprocessing
│   ├── detection/               # Video processing and fashion detection
│   ├── matching/                # Product matching with CLIP+FAISS
│   ├── classification/          # Vibe classification with NLP
│   ├── config.py               # Configuration settings
│   └── pipeline.py             # Main pipeline orchestration
├── outputs/                     # Evaluation JSONs (per video)
├── requirements.txt            # Python dependencies
├── main.py                     # MAIN ENTRY POINT
└── README.md                   # Setup + instructions
```

## 🛠 Technical Implementation

### Core Components
1. **Object Detection**: YOLOv8 for person and fashion item detection
2. **Product Matching**: CLIP embeddings + FAISS similarity search
3. **Vibe Classification**: DistilBERT NLP for aesthetic categorization
4. **Video Processing**: Scene change detection for optimal frame extraction

### Performance
- **Processing Speed**: ~20-30 seconds per video
- **Detection Accuracy**: Optimized for fashion content
- **Scalability**: Handles 11K+ product catalog efficiently

## 📊 Evaluation JSONs (Per Video)

The system generates evaluation JSON files for each video in the `/outputs` directory with the **exact hackathon-required structure**:

```json
{
  "video_id": "2025-05-28_13-40-09_UTC",
  "vibes": ["Cottagecore"],
  "products": [
    {
      "type": "shorts",
      "color": "white",
      "matched_product_id": "14981",
      "match_type": "similar",
      "confidence": 0.757
    }
  ]
}
```

### Output Files Generated:
- `outputs/2025-05-22_08-25-12_UTC_analysis.json`
- `outputs/2025-05-27_13-46-16_UTC_analysis.json`
- `outputs/2025-05-28_13-40-09_UTC_analysis.json`
- `outputs/2025-05-28_13-42-32_UTC_analysis.json`
- `outputs/2025-05-31_14-01-37_UTC_analysis.json`
- `outputs/2025-06-02_11-31-19_UTC_analysis.json`
- `outputs/final_results.json` (combined results)

## 🎨 Supported Fashion Vibes

The system classifies content into these 7 aesthetic categories:

- **Coquette**: Feminine, romantic, soft, delicate
- **Clean Girl**: Minimal, natural, effortless
- **Cottagecore**: Rustic, vintage, nature-inspired
- **Streetcore**: Urban, edgy, contemporary
- **Y2K**: Futuristic, metallic, 2000s-inspired
- **Boho**: Bohemian, free-spirited, artistic
- **Party Glam**: Glamorous, sparkly, elegant

## 👗 Fashion Item Detection

Detects and classifies:
- **Tops**: Shirts, blouses, sweaters, jackets
- **Bottoms**: Pants, jeans, skirts, shorts
- **Dresses**: Full-body garments
- **Accessories**: Bags, jewelry, hats
- **Footwear**: Shoes, boots, sneakers

## 📈 System Performance

- **Processing Speed**: ~20-30 seconds per video
- **Detection Rate**: 96 persons detected across 98 frames (98% success rate)
- **Fashion Items**: 227 items detected from person regions
- **Product Matching**: CLIP embeddings with FAISS similarity search
- **Scalability**: Handles 11K+ product catalog efficiently

## 🔧 Technical Stack

- **YOLOv8**: Object detection via ultralytics package
- **CLIP**: OpenAI/HuggingFace image embeddings
- **FAISS**: Facebook AI similarity search
- **DistilBERT**: Transformer-based NLP classification
- **spaCy**: Natural language processing

---

**🏆 Built for the Flickd AI Hackathon**
*Complete AI-powered fashion video analysis system*
