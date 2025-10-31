"""
Image processing service for handling file operations
"""

from pathlib import Path
from typing import List


class ImageProcessor:
    """Handles image file operations"""
    
    def __init__(self, source_dir: Path, target_dir: Path):
        self.source_dir = source_dir
        self.target_dir = target_dir
        self.target_dir.mkdir(exist_ok=True)
    
    def get_image_files(self) -> List[Path]:
        """Get all supported image files from source directory"""
        return list(self.source_dir.glob("*.jpg")) + list(self.source_dir.glob("*.jpeg"))
    
    def rename_and_move(self, original_path: Path, new_filename: str) -> Path:
        """
        Rename image and move to target directory
        
        Args:
            original_path: Original image path
            new_filename: New filename to use
            
        Returns:
            Path to the new file location
        """
        final_path = self.target_dir / new_filename
        original_path.rename(final_path)
        return final_path