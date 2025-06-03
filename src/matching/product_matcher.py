"""
Product matching using CLIP embeddings and FAISS similarity search
"""
import torch
import numpy as np
import faiss
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from typing import List, Dict, Tuple, Optional
import pickle
import logging
from pathlib import Path
import pandas as pd

from ..config import (
    CLIP_MODEL, SIMILARITY_THRESHOLDS, MODELS_DIR,
    PROCESSED_DIR, IMAGE_SIZE
)

logger = logging.getLogger(__name__)


class ProductMatcher:
    """Matches detected fashion items to product catalog using CLIP embeddings"""
    
    def __init__(self):
        self.clip_model = None
        self.clip_processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.faiss_index = None
        self.product_embeddings = None
        self.product_ids = []
        self.product_metadata = {}
        
        self.load_models()
    
    def load_models(self):
        """Load CLIP model and processor"""
        try:
            self.clip_model = CLIPModel.from_pretrained(CLIP_MODEL)
            self.clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
            self.clip_model.to(self.device)
            self.clip_model.eval()
            logger.info(f"Loaded CLIP model on {self.device}")
        except Exception as e:
            logger.error(f"Error loading CLIP model: {e}")
    
    def generate_image_embedding(self, image: Image.Image) -> np.ndarray:
        """Generate CLIP embedding for an image with timeout and error handling"""
        try:
            # Validate image
            if image is None or image.size[0] == 0 or image.size[1] == 0:
                logger.warning("Invalid image provided for embedding")
                return np.zeros(512)

            # Resize image to prevent memory issues
            max_size = 224
            if image.size[0] > max_size or image.size[1] > max_size:
                image = image.resize((max_size, max_size), Image.Resampling.LANCZOS)

            # Preprocess image with error handling
            try:
                inputs = self.clip_processor(images=image, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            except Exception as e:
                logger.error(f"Error preprocessing image: {e}")
                return np.zeros(512)

            # Generate embedding with timeout protection
            try:
                with torch.no_grad():
                    # Clear cache to prevent memory buildup
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    image_features = self.clip_model.get_image_features(**inputs)

                    # Check if features are valid
                    if image_features is None or torch.isnan(image_features).any():
                        logger.warning("Invalid features generated")
                        return np.zeros(512)

                    # Normalize the features
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                    # Convert to numpy
                    result = image_features.cpu().numpy().flatten()

                    # Validate result
                    if len(result) == 0 or np.isnan(result).any():
                        logger.warning("Invalid embedding result")
                        return np.zeros(512)

                    return result

            except KeyboardInterrupt:
                logger.error("CLIP processing interrupted by user")
                raise
            except Exception as e:
                logger.error(f"Error during CLIP inference: {e}")
                return np.zeros(512)

        except KeyboardInterrupt:
            logger.error("Image embedding generation interrupted")
            raise
        except Exception as e:
            logger.error(f"Error generating image embedding: {e}")
            return np.zeros(512)
    
    def build_product_index(self, product_images: Dict[str, str], force_rebuild: bool = False):
        """Build FAISS index from product images"""
        index_path = MODELS_DIR / "product_index.faiss"
        embeddings_path = MODELS_DIR / "product_embeddings.pkl"
        metadata_path = MODELS_DIR / "product_metadata.pkl"
        
        # Check if index already exists
        if not force_rebuild and index_path.exists() and embeddings_path.exists():
            logger.info("Loading existing product index...")
            self.load_product_index()
            return
        
        logger.info(f"Building product index for {len(product_images)} products...")
        
        embeddings = []
        valid_product_ids = []
        metadata = {}
        
        for product_id, image_path in product_images.items():
            try:
                # Load and process image
                image = Image.open(image_path).convert('RGB')
                image = image.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
                
                # Generate embedding
                embedding = self.generate_image_embedding(image)
                
                if embedding is not None and not np.all(embedding == 0):
                    embeddings.append(embedding)
                    valid_product_ids.append(product_id)
                    metadata[product_id] = {
                        "image_path": image_path,
                        "embedding_index": len(embeddings) - 1
                    }
                
            except Exception as e:
                logger.error(f"Error processing product {product_id}: {e}")
                continue
        
        if not embeddings:
            logger.error("No valid embeddings generated")
            return
        
        # Convert to numpy array
        embeddings_array = np.array(embeddings).astype('float32')
        
        # Build FAISS index
        dimension = embeddings_array.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        self.faiss_index.add(embeddings_array)
        
        # Store data
        self.product_embeddings = embeddings_array
        self.product_ids = valid_product_ids
        self.product_metadata = metadata
        
        # Save to disk
        faiss.write_index(self.faiss_index, str(index_path))
        
        with open(embeddings_path, 'wb') as f:
            pickle.dump({
                'embeddings': embeddings_array,
                'product_ids': valid_product_ids
            }, f)
        
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        logger.info(f"Built and saved product index with {len(valid_product_ids)} products")
    
    def load_product_index(self):
        """Load existing product index from disk"""
        try:
            index_path = MODELS_DIR / "product_index.faiss"
            embeddings_path = MODELS_DIR / "product_embeddings.pkl"
            metadata_path = MODELS_DIR / "product_metadata.pkl"
            
            # Load FAISS index
            self.faiss_index = faiss.read_index(str(index_path))
            
            # Load embeddings and product IDs
            with open(embeddings_path, 'rb') as f:
                data = pickle.load(f)
                self.product_embeddings = data['embeddings']
                self.product_ids = data['product_ids']
            
            # Load metadata
            with open(metadata_path, 'rb') as f:
                self.product_metadata = pickle.load(f)
            
            logger.info(f"Loaded product index with {len(self.product_ids)} products")
            
        except Exception as e:
            logger.error(f"Error loading product index: {e}")
    
    def find_similar_products(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        """Find similar products using FAISS search"""
        if self.faiss_index is None:
            logger.error("Product index not loaded")
            return []
        
        try:
            # Normalize query embedding
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
            query_embedding = query_embedding.reshape(1, -1).astype('float32')
            
            # Search
            similarities, indices = self.faiss_index.search(query_embedding, top_k)
            
            results = []
            for similarity, idx in zip(similarities[0], indices[0]):
                if idx < len(self.product_ids):
                    product_id = self.product_ids[idx]
                    results.append((product_id, float(similarity)))
            
            return results
            
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            return []
    
    def match_fashion_item(self, item_image: Image.Image, item_info: Dict) -> Optional[Dict]:
        """Match a detected fashion item to products in catalog with robust error handling"""
        try:
            # Validate inputs
            if item_image is None:
                logger.warning("No image provided for matching")
                return None

            if item_image.size[0] == 0 or item_image.size[1] == 0:
                logger.warning("Invalid image size for matching")
                return None

            # Generate embedding for the detected item with timeout protection
            try:
                item_embedding = self.generate_image_embedding(item_image)
            except KeyboardInterrupt:
                logger.error("Fashion item matching interrupted by user")
                raise
            except Exception as e:
                logger.error(f"Failed to generate embedding for fashion item: {e}")
                return None

            if item_embedding is None or np.all(item_embedding == 0):
                logger.warning("Failed to generate valid embedding for fashion item")
                return None

            # Find similar products with error handling
            try:
                similar_products = self.find_similar_products(item_embedding, top_k=10)
            except Exception as e:
                logger.error(f"Error finding similar products: {e}")
                return None

            if not similar_products:
                logger.debug("No similar products found")
                return None

            # Get the best match
            best_product_id, best_similarity = similar_products[0]

            # Determine match type based on similarity
            if best_similarity >= SIMILARITY_THRESHOLDS["exact"]:
                match_type = "exact"
            elif best_similarity >= SIMILARITY_THRESHOLDS["similar"]:
                match_type = "similar"
            else:
                logger.debug(f"Similarity {best_similarity} below threshold")
                return None  # Below minimum threshold

            match_result = {
                "type": item_info.get("type", "unknown"),
                "color": item_info.get("color", "unknown"),
                "matched_product_id": best_product_id,
                "match_type": match_type,
                "confidence": round(best_similarity, 3),
                "similar_products": [
                    {"product_id": pid, "similarity": round(sim, 3)}
                    for pid, sim in similar_products[:5]
                ]
            }

            return match_result

        except KeyboardInterrupt:
            logger.error("Fashion item matching interrupted")
            raise
        except Exception as e:
            logger.error(f"Error matching fashion item: {e}")
            return None
    
    def match_detected_items(self, detected_items: List[Dict], frame_image: Image.Image) -> List[Dict]:
        """Match all detected items in a frame"""
        matched_items = []

        for item in detected_items:
            try:
                # Extract item region from frame
                bbox = item.get("bbox", [0, 0, frame_image.width, frame_image.height])

                # Handle both (x, y, w, h) and (x1, y1, x2, y2) formats
                if len(bbox) == 4:
                    x, y, w, h = bbox
                    # Convert (x, y, w, h) to (x1, y1, x2, y2) for cropping
                    x1, y1, x2, y2 = x, y, x + w, y + h
                else:
                    x1, y1, x2, y2 = bbox

                # Ensure coordinates are within image bounds
                x1 = max(0, min(x1, frame_image.width))
                y1 = max(0, min(y1, frame_image.height))
                x2 = max(x1, min(x2, frame_image.width))
                y2 = max(y1, min(y2, frame_image.height))

                # Crop item region
                item_crop = frame_image.crop((x1, y1, x2, y2))

                # Match the item
                match_result = self.match_fashion_item(item_crop, item)

                if match_result:
                    matched_items.append(match_result)

            except Exception as e:
                logger.error(f"Error matching detected item: {e}")
                continue

        return matched_items
