"""
Search interface for querying and formatting results
"""

from pathlib import Path
from typing import List, Tuple, Optional

from .chroma_service import ChromaDBService
from .config import Config


class ImageSearchInterface:
    """Handles search queries and result formatting"""
    
    def __init__(self, db_service: ChromaDBService, images_dir: Path):
        self.db_service = db_service
        self.images_dir = images_dir
    
    def search(self, query: str) -> Tuple[List[Tuple[str, str]], str]:
        """
        Search for images matching the query
        
        Args:
            query: Search query text
            
        Returns:
            Tuple of (image list, status message)
            Image list contains tuples of (image_path, caption)
        """
        if not query.strip():
            return [], "Please enter a search query"
        
        print(f"\n{'='*60}")
        print(f"Search query: {query}")
        print(f"{'='*60}")
        
        # Perform search
        results = self.db_service.search(query)
        
        # Filter and format results
        images = []
        filtered_docs = self._filter_by_distance(results)
        
        for doc in filtered_docs:
            image_path = self._get_image_path(doc['id'])
            if image_path and image_path.exists():
                caption = self._format_caption(doc)
                images.append((str(image_path), caption))
        
        # Generate status message
        if images:
            status = f"✅ Found {len(images)} matching images"
        else:
            status = "❌ No matching images found. Try a different search term."
        
        print(f"Results: {len(images)} images\n")
        return images, status
    
    def _filter_by_distance(self, results: dict) -> List[dict]:
        """Filter results by distance threshold"""
        filtered = []
        
        print("\nDistance scores:")
        for i, distance in enumerate(results['distances'][0]):
            doc_preview = results['documents'][0][i][:60]
            print(f"  {i+1}. Distance: {distance:.4f} - {doc_preview}...")
            
            if distance < Config.DISTANCE_THRESHOLD:
                filtered.append({
                    'id': results['ids'][0][i],
                    'document': results['documents'][0][i],
                    'distance': distance
                })
        
        return filtered
    
    def _get_image_path(self, embedding_id: str) -> Optional[Path]:
        """Convert embedding ID to image path"""
        image_name = embedding_id.replace('emb-', '') + "_rn.jpg"
        return self.images_dir / image_name
    
    def _format_caption(self, doc: dict) -> str:
        """Format caption for display"""
        similarity_pct = (1 - doc['distance']) * 100
        return f"{doc['document'][:80]}\n📊 Similarity: {similarity_pct:.1f}%"