"""
Main pipeline for processing videos and generating fashion analysis results
"""
import json
import logging
import time
import numpy as np
from pathlib import Path
from typing import Dict, List
from PIL import Image

from .config import OUTPUTS_DIR
from .preprocessing.data_loader import DataLoader
from .detection.video_processor import VideoProcessor
from .detection.fashion_detector import FashionDetector
from .matching.product_matcher import ProductMatcher
from .classification.vibe_classifier import VibeClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FashionAnalysisPipeline:
    """Main pipeline for fashion video analysis"""
    
    def __init__(self):
        self.data_loader = DataLoader()
        self.video_processor = VideoProcessor()
        self.fashion_detector = FashionDetector()
        self.product_matcher = ProductMatcher()
        self.vibe_classifier = VibeClassifier()
        
        self.setup_complete = False
    
    def setup(self, cache_images: bool = True, image_limit: int = None):
        """Setup the pipeline by loading data and building indices"""
        logger.info("Setting up Fashion Analysis Pipeline...")
        
        try:
            # Load all data
            logger.info("Loading data...")
            self.data_loader.load_product_data()
            self.data_loader.load_images_data()
            self.data_loader.load_vibes_list()
            
            # Cache product images if requested
            if cache_images:
                logger.info("Caching product images...")
                cached_images = self.data_loader.cache_all_product_images(limit=image_limit)
                
                # Build product matching index
                logger.info("Building product matching index...")
                self.product_matcher.build_product_index(cached_images)
            
            self.setup_complete = True
            logger.info("Pipeline setup complete!")
            
        except Exception as e:
            logger.error(f"Error during pipeline setup: {e}")
            raise
    
    def process_video(self, video_id: str, video_path: str = None) -> Dict:
        """Process a single video and return analysis results

        Args:
            video_id: Identifier for the video (used for output naming)
            video_path: Optional custom path to video file. If None, uses default location.
        """
        if not self.setup_complete:
            raise RuntimeError("Pipeline not set up. Call setup() first.")

        logger.info(f"Processing video: {video_id}")

        try:
            # Load video metadata (may not exist for new videos)
            metadata = self.data_loader.load_video_metadata(video_id)

            # Extract frames from video (supports custom paths)
            if video_path:
                frames = self.video_processor.extract_frames_from_path(video_path)
            else:
                frames = self.video_processor.extract_frames_at_scenes(video_id)

            if not frames:
                logger.warning(f"No frames extracted from {video_id}")
                return self._create_empty_result(video_id)

            # Process each frame for fashion detection
            all_detected_items = []
            all_matched_products = []
            total_persons_detected = 0

            for i, frame in enumerate(frames):
                try:
                    logger.info(f"Processing frame {i+1}/{len(frames)}")

                    # Convert frame to PIL Image but DON'T resize for person detection
                    if isinstance(frame, np.ndarray):
                        frame_image = Image.fromarray(frame)
                    else:
                        frame_image = frame

                    # Validate frame image
                    if frame_image is None or frame_image.size[0] == 0 or frame_image.size[1] == 0:
                        logger.warning(f"Invalid frame {i+1}, skipping")
                        continue

                    # Detect fashion items with frame number
                    try:
                        detected_items = self.fashion_detector.detect_fashion_items(frame_image, frame_number=i)
                    except KeyboardInterrupt:
                        logger.error("Frame processing interrupted by user")
                        raise
                    except Exception as e:
                        logger.error(f"Error detecting fashion items in frame {i+1}: {e}")
                        continue

                    # Count unique persons detected in this frame
                    if detected_items:
                        frame_persons = set()
                        for item in detected_items:
                            person_id = item.get('person_id')
                            if person_id is not None:
                                frame_persons.add(person_id)
                        total_persons_detected += len(frame_persons)

                    if detected_items:
                        # Match detected items to products with error handling
                        try:
                            matched_items = self.product_matcher.match_detected_items(
                                detected_items, frame_image
                            )
                            all_detected_items.extend(detected_items)
                            all_matched_products.extend(matched_items)
                        except KeyboardInterrupt:
                            logger.error("Product matching interrupted by user")
                            raise
                        except Exception as e:
                            logger.error(f"Error matching products in frame {i+1}: {e}")
                            # Still add detected items even if matching fails
                            all_detected_items.extend(detected_items)
                            continue

                except KeyboardInterrupt:
                    logger.error("Video processing interrupted by user")
                    raise
                except Exception as e:
                    logger.error(f"Error processing frame {i+1}: {e}")
                    continue

            # Classify vibes (gracefully handle missing metadata)
            vibes = self.vibe_classifier.classify_vibes(metadata) if metadata else []

            # Aggregate and deduplicate results
            final_products = self._aggregate_product_matches(all_matched_products)

            # Create result
            result = {
                "video_id": video_id,
                "vibes": vibes,
                "products": final_products,
                "metadata": {
                    "frames_processed": len(frames),
                    "items_detected": len(all_detected_items),
                    "persons_detected": total_persons_detected,
                    "products_matched": len(all_matched_products),
                    "final_products": len(final_products),
                    "has_metadata": bool(metadata)
                }
            }

            logger.info(f"Completed processing {video_id}: {len(vibes)} vibes, {len(final_products)} products")
            return result

        except Exception as e:
            logger.error(f"Error processing video {video_id}: {e}")
            return self._create_empty_result(video_id, error=str(e))
    
    def _aggregate_product_matches(self, matched_products: List[Dict]) -> List[Dict]:
        """Aggregate and deduplicate product matches"""
        if not matched_products:
            return []
        
        # Group by product ID
        product_groups = {}
        
        for product in matched_products:
            product_id = product.get("matched_product_id")
            if product_id:
                if product_id not in product_groups:
                    product_groups[product_id] = []
                product_groups[product_id].append(product)
        
        # Aggregate each group
        final_products = []
        
        for product_id, products in product_groups.items():
            # Take the best match (highest confidence)
            best_match = max(products, key=lambda x: x.get("confidence", 0))
            
            # Calculate average confidence
            avg_confidence = sum(p.get("confidence", 0) for p in products) / len(products)
            
            # Create final product entry
            final_product = {
                "type": best_match.get("type", "unknown"),
                "color": best_match.get("color", "unknown"),
                "matched_product_id": product_id,
                "match_type": best_match.get("match_type", "similar"),
                "confidence": round(avg_confidence, 3)
            }
            
            final_products.append(final_product)
        
        # Sort by confidence
        final_products.sort(key=lambda x: x["confidence"], reverse=True)
        
        return final_products
    
    def _create_empty_result(self, video_id: str, error: str = None) -> Dict:
        """Create empty result structure"""
        result = {
            "video_id": video_id,
            "vibes": [],
            "products": []
        }
        
        if error:
            result["error"] = error
        
        return result
    
    def process_all_videos(self) -> Dict[str, Dict]:
        """Process all available videos with error handling"""
        video_ids = self.data_loader.get_video_list()
        results = {}

        logger.info(f"Processing {len(video_ids)} videos...")

        for i, video_id in enumerate(video_ids, 1):
            try:
                logger.info(f"Processing video {i}/{len(video_ids)}: {video_id}")

                # Process video with timeout protection
                video_start_time = time.time()
                result = self.process_video(video_id)
                video_time = time.time() - video_start_time

                logger.info(f"Completed {video_id} in {video_time:.2f} seconds")

                results[video_id] = result

                # Save individual result
                self.save_result(result)

            except KeyboardInterrupt:
                logger.error(f"Processing interrupted by user at video {video_id}")
                results[video_id] = self._create_empty_result(video_id, error="Interrupted by user")
                break  # Stop processing remaining videos
            except Exception as e:
                logger.error(f"Failed to process {video_id}: {e}")
                results[video_id] = self._create_empty_result(video_id, error=str(e))
                continue  # Continue with next video

        return results
    
    def save_result(self, result: Dict):
        """Save result to JSON file"""
        try:
            video_id = result["video_id"]
            output_file = OUTPUTS_DIR / f"{video_id}_analysis.json"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved result to {output_file}")
            
        except Exception as e:
            logger.error(f"Error saving result: {e}")
    
    def save_all_results(self, results: Dict[str, Dict]):
        """Save all results to a combined file"""
        try:
            output_file = OUTPUTS_DIR / "all_results.json"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved all results to {output_file}")
            
        except Exception as e:
            logger.error(f"Error saving all results: {e}")
    
    def get_pipeline_stats(self) -> Dict:
        """Get pipeline statistics"""
        stats = {
            "setup_complete": self.setup_complete,
            "available_videos": len(self.data_loader.get_video_list()),
            "product_catalog_size": len(self.data_loader.images_data) if self.data_loader.images_data is not None else 0,
            "available_vibes": len(self.data_loader.vibes_list) if self.data_loader.vibes_list else 0
        }
        
        if hasattr(self.product_matcher, 'product_ids'):
            stats["indexed_products"] = len(self.product_matcher.product_ids)
        
        return stats
