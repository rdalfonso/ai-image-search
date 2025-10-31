"""
Image indexing pipeline for processing and storing images
"""

from pathlib import Path

from .vision_service import VisionModelService
from .image_processor import ImageProcessor
from .chroma_service import ChromaDBService


class ImageIndexer:
    """Orchestrates the image indexing process"""
    
    def __init__(
        self,
        vision_service: VisionModelService,
        image_processor: ImageProcessor,
        db_service: ChromaDBService
    ):
        self.vision_service = vision_service
        self.image_processor = image_processor
        self.db_service = db_service
    
    def index_all_images(self) -> int:
        """
        Process and index all images in the source directory
        
        Returns:
            Number of successfully indexed images
        """
        image_files = self.image_processor.get_image_files()
        indexed_count = 0
        
        print(f"Found {len(image_files)} images to process")
        
        for image_path in image_files:
            if self._index_single_image(image_path):
                indexed_count += 1
        
        print(f"\nIndexing complete: {indexed_count}/{len(image_files)} images processed")
        return indexed_count
    
    def _index_single_image(self, image_path: Path) -> bool:
        """
        Process and index a single image
        
        Returns:
            True if successful, False otherwise
        """
        print(f"\nProcessing: {image_path.name}")
        
        # Get description from vision model
        description = self.vision_service.describe_image(image_path)
        if not description:
            return False
        
        # Rename and move file
        new_filename = description.get_safe_filename()
        try:
            new_path = self.image_processor.rename_and_move(image_path, new_filename)
            print(f"Renamed to: {new_filename}")
        except Exception as e:
            print(f"Error renaming file: {e}")
            return False
        
        # Add to database
        embedding_id = description.get_embedding_id()
        self.db_service.add_document(embedding_id, description.description)
        print(f"Description: {description.description}")
        
        return True