"""
Image Description and Search System
Processes images with vision model, stores embeddings in ChromaDB, and provides search interface
"""

import sys
from pathlib import Path

# Add the parent directory to the Python path to import modules
sys.path.append(str(Path(__file__).parent.parent))

from modules import (
    Config,
    VisionModelService,
    ImageProcessor,
    ChromaDBService,
    ImageIndexer,
    ImageSearchInterface,
    create_gradio_interface
)


# ============================================================================
# Main Application
# ============================================================================

def main():
    """Main application entry point"""
    
    # Initialize services
    vision_service = VisionModelService()
    image_processor = ImageProcessor(Config.IMAGES_DIR, Config.RENAMED_DIR)
    db_service = ChromaDBService()
    
    # Index images (only if there are unprocessed images)
    if Config.IMAGES_DIR.exists() and list(Config.IMAGES_DIR.glob("*.jpg")):
        print("Starting image indexing...")
        indexer = ImageIndexer(vision_service, image_processor, db_service)
        indexer.index_all_images()
    else:
        print("No new images to index")
    
    print(f"\nTotal documents in database: {db_service.get_collection_count()}")
    
    # Create and launch search interface
    search_interface = ImageSearchInterface(db_service, Config.RENAMED_DIR)
    demo = create_gradio_interface(search_interface)
    
    print("\nLaunching Gradio interface...")
    demo.launch()


if __name__ == "__main__":
    main()