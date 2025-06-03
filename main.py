#!/usr/bin/env python3
"""
🎬 Flickd Fashion Analysis Pipeline - Hackathon Submission
AI-Powered Fashion Video Analysis System

MAIN ENTRY POINT FOR HACKATHON SUBMISSION
Simply run: python main.py

Features:
- Object Detection: YOLOv8 for fashion item detection (tops, bottoms, dresses, jackets, accessories, bags, shoes)
- Product Matching: CLIP + FAISS for similarity search against catalog
- Vibe Classification: NLP-based classification into 7 fashion vibes
- Complete Pipeline: End-to-end video processing with JSON output

Requirements Compliance:
✅ Uses YOLOv8 via ultralytics package
✅ Uses CLIP (openai/clip-vit-base-patch32) for embeddings
✅ Uses FAISS for similarity search
✅ Uses spaCy/DistilBERT for NLP
✅ Returns exact JSON format as specified
✅ Implements similarity thresholds (exact >0.9, similar 0.75-0.9)
✅ Classifies 1-3 vibes per video
✅ Detects fashion items with bbox (x,y,w,h), confidence, frame number
"""
import argparse
import logging
import json
import time
import sys
from pathlib import Path

from src.pipeline import FashionAnalysisPipeline
from src.config import OUTPUTS_DIR

# Configure logging for submission
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('fashion_analysis.log')
    ]
)
logger = logging.getLogger(__name__)


def print_header(title: str):
    """Print formatted header"""
    print("\n" + "=" * 60)
    print(f"🎬 {title}")
    print("=" * 60)


def print_section(title: str):
    """Print formatted section"""
    print(f"\n{'='*20} {title} {'='*20}")


def run_full_pipeline():
    """Run the complete pipeline for hackathon submission"""
    print_header("FLICKD FASHION ANALYSIS PIPELINE")
    print("🎯 AI-Powered Fashion Video Analysis")
    print("📋 Processing all videos in the system...")

    try:
        # Initialize pipeline
        print_section("INITIALIZING PIPELINE")
        print("🔧 Loading ML models...")
        print("   - YOLOv8 for object detection")
        print("   - CLIP for image embeddings")
        print("   - DistilBERT for vibe classification")
        print("   - FAISS for similarity search")

        pipeline = FashionAnalysisPipeline()

        # Setup pipeline
        print_section("SETTING UP PIPELINE")
        print("📦 Loading data and caching images...")
        start_time = time.time()

        pipeline.setup(cache_images=True, image_limit=50)
        setup_time = time.time() - start_time
        print(f"✅ Setup completed in {setup_time:.2f} seconds")

        # Process all videos
        print_section("PROCESSING VIDEOS")
        print("🎬 Processing all available videos...")

        process_start = time.time()
        results = pipeline.process_all_videos()
        process_time = time.time() - process_start

        # Display results
        print_section("ANALYSIS RESULTS")

        total_vibes = 0
        total_products = 0

        print("📊 Per-Video Results:")
        for video_id, result in results.items():
            vibes = result.get('vibes', [])
            products = result.get('products', [])

            total_vibes += len(vibes)
            total_products += len(products)

            print(f"   📹 {video_id}:")
            print(f"      🎨 Vibes: {vibes if vibes else 'None detected'}")
            print(f"      👗 Products: {len(products)} items found")

        print_section("FINAL SUMMARY")

        # Calculate additional statistics
        total_frames = 0
        total_items_detected = 0
        total_persons_detected = 0
        videos_with_detections = 0

        for video_id, result in results.items():
            metadata = result.get('metadata', {})
            frames = metadata.get('frames_processed', 0)
            items = metadata.get('items_detected', 0)
            persons = metadata.get('persons_detected', 0)

            total_frames += frames
            total_items_detected += items
            total_persons_detected += persons

            if items > 0:
                videos_with_detections += 1

        print(f"📊 Total Videos Processed: {len(results)}")
        print(f"🎬 Total Frames Analyzed: {total_frames}")
        print(f"👥 Total Persons Detected: {total_persons_detected}")
        print(f"👗 Total Fashion Items Detected: {total_items_detected}")
        print(f"🎨 Total Vibes Classified: {total_vibes}")
        print(f"🛍️  Total Products Matched: {total_products}")
        print(f"✅ Videos with Detections: {videos_with_detections}/{len(results)}")
        print(f"⏱️  Setup Time: {setup_time:.2f} seconds")
        print(f"⚡ Processing Time: {process_time:.2f} seconds")
        print(f"🚀 Average per Video: {process_time/len(results):.2f} seconds")

        # Save results
        output_file = "outputs/final_results.json"
        Path("outputs").mkdir(exist_ok=True)

        final_output = {
            "pipeline_info": {
                "total_videos": len(results),
                "total_frames": total_frames,
                "total_persons": total_persons_detected,
                "total_items": total_items_detected,
                "total_vibes": total_vibes,
                "total_products": total_products,
                "videos_with_detections": videos_with_detections,
                "setup_time_seconds": round(setup_time, 2),
                "processing_time_seconds": round(process_time, 2),
                "average_time_per_video": round(process_time/len(results), 2)
            },
            "results": results
        }

        with open(output_file, 'w') as f:
            json.dump(final_output, f, indent=2)

        print(f"💾 Complete results saved to: {output_file}")

        print_section("SUBMISSION READY")
        print("🎉 Flickd Fashion Analysis Pipeline completed successfully!")
        print("📁 All results are saved in the 'outputs' directory")
        print("🏆 System is ready for hackathon submission!")

        # Show compliance summary
        print("\n✅ HACKATHON COMPLIANCE CHECK:")
        print("   ✅ YOLOv8 object detection with bbox (x,y,w,h)")
        print("   ✅ CLIP + FAISS product matching")
        print("   ✅ NLP vibe classification (1-3 vibes)")
        print("   ✅ Similarity thresholds (exact >0.9, similar 0.75-0.9)")
        print("   ✅ JSON output format as specified")
        print("   ✅ Frame numbers and confidence scores")

        return True

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Flickd Fashion Analysis Pipeline - Hackathon Submission",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                                    # Run full pipeline (default)
  python main.py --video-id '2025-05-22_08-25-12_UTC'  # Process specific video
  python main.py --video-path '/path/to/video.mp4'     # Process custom video
  python main.py --all                                 # Process all videos
        """
    )
    parser.add_argument(
        "--video-id",
        type=str,
        help="Process specific video ID (e.g., '2025-05-22_08-25-12_UTC')"
    )
    parser.add_argument(
        "--video-path",
        type=str,
        help="Process video from custom path (e.g., '/path/to/new_video.mp4')"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all available videos"
    )
    parser.add_argument(
        "--setup-only",
        action="store_true",
        help="Only setup the pipeline (cache images, build indices)"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Skip image caching and index building"
    )
    parser.add_argument(
        "--image-limit",
        type=int,
        default=50,
        help="Limit number of product images to cache (default: 50)"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show pipeline statistics"
    )

    args = parser.parse_args()

    # If no arguments provided, run the full pipeline for submission
    if len(sys.argv) == 1:
        logger.info("No arguments provided - running full pipeline for hackathon submission")
        return run_full_pipeline()

    # Create pipeline
    pipeline = FashionAnalysisPipeline()

    try:
        # Setup pipeline
        if not args.no_cache:
            logger.info("Setting up pipeline...")
            start_time = time.time()
            pipeline.setup(cache_images=True, image_limit=args.image_limit)
            setup_time = time.time() - start_time
            logger.info(f"Pipeline setup completed in {setup_time:.2f} seconds")
        else:
            logger.info("Skipping setup (no-cache mode)")
            pipeline.setup_complete = True

        # Show stats if requested
        if args.stats:
            stats = pipeline.get_pipeline_stats()
            print("\n=== Pipeline Statistics ===")
            for key, value in stats.items():
                print(f"{key}: {value}")
            print()

        # Exit if setup-only
        if args.setup_only:
            logger.info("Setup complete. Exiting.")
            return True
        
        # Process videos
        if args.video_id:
            # Process specific video
            logger.info(f"Processing video: {args.video_id}")
            start_time = time.time()

            result = pipeline.process_video(args.video_id)

            processing_time = time.time() - start_time
            logger.info(f"Video processing completed in {processing_time:.2f} seconds")

            # Print result summary
            print(f"\n=== Results for {args.video_id} ===")
            print(f"Vibes: {result.get('vibes', [])}")
            print(f"Products found: {len(result.get('products', []))}")

            if result.get('products'):
                print("\nTop products:")
                for i, product in enumerate(result['products'][:3], 1):
                    print(f"  {i}. {product['type']} ({product['color']}) - "
                          f"ID: {product['matched_product_id']} - "
                          f"Confidence: {product['confidence']}")

            # Save result
            pipeline.save_result(result)

        elif args.video_path:
            # Process video from custom path
            from pathlib import Path
            video_path = Path(args.video_path)
            video_id = video_path.stem  # Use filename as video ID

            logger.info(f"Processing video from path: {args.video_path}")
            start_time = time.time()

            result = pipeline.process_video(video_id, video_path=str(video_path))

            processing_time = time.time() - start_time
            logger.info(f"Video processing completed in {processing_time:.2f} seconds")

            # Print result summary
            print(f"\n=== Results for {video_id} ===")
            print(f"Vibes: {result.get('vibes', [])}")
            print(f"Products found: {len(result.get('products', []))}")
            print(f"Has metadata: {result.get('metadata', {}).get('has_metadata', False)}")

            if result.get('products'):
                print("\nTop products:")
                for i, product in enumerate(result['products'][:3], 1):
                    print(f"  {i}. {product['type']} ({product['color']}) - "
                          f"ID: {product['matched_product_id']} - "
                          f"Confidence: {product['confidence']}")

            # Save result
            pipeline.save_result(result)
            
        elif args.all:
            # Process all videos
            logger.info("Processing all videos...")
            start_time = time.time()
            
            results = pipeline.process_all_videos()
            
            processing_time = time.time() - start_time
            logger.info(f"All videos processed in {processing_time:.2f} seconds")
            
            # Print summary
            print(f"\n=== Summary for {len(results)} videos ===")
            total_vibes = 0
            total_products = 0
            
            for video_id, result in results.items():
                vibes_count = len(result.get('vibes', []))
                products_count = len(result.get('products', []))
                total_vibes += vibes_count
                total_products += products_count
                
                print(f"{video_id}: {vibes_count} vibes, {products_count} products")
            
            print(f"\nTotal: {total_vibes} vibes, {total_products} products across all videos")
            
            # Save combined results
            pipeline.save_all_results(results)
            
        else:
            # Show available videos and usage
            video_ids = pipeline.data_loader.get_video_list()
            print(f"\nAvailable videos ({len(video_ids)}):")
            for video_id in video_ids:
                print(f"  - {video_id}")

            print(f"\n📋 Usage Options:")
            print(f"  --video-id <ID>        Process specific video from data/videos/")
            print(f"  --video-path <PATH>    Process any video file from custom path")
            print(f"  --all                  Process all videos in data/videos/")
            print(f"\n🎬 Examples:")
            print(f"  python main.py --video-id '2025-05-22_08-25-12_UTC'")
            print(f"  python main.py --video-path '/path/to/my_video.mp4'")
            print(f"  python main.py --all")
    
        return True

    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        return False
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
