"""
Pydantic models for the image recognition system
"""

import re
from pydantic import BaseModel, Field


class ImageDescription(BaseModel):
    """Structured output from vision model"""
    description: str = Field(..., min_length=1, description="A short description of the image")
    name: str = Field(..., min_length=1, max_length=50, description="A few words for filename (no spaces)")
    
    def get_safe_filename(self) -> str:
        """Generate a safe filename from the name field"""
        safe_name = re.sub(r'[^\w\s-]', '', self.name).strip()
        return f"{safe_name.replace(' ', '_').lower()}_rn.jpg"
    
    def get_embedding_id(self) -> str:
        """Generate consistent embedding ID"""
        safe_name = re.sub(r'[^\w\s-]', '', self.name).strip()
        return f"emb-{safe_name.replace(' ', '_').lower()}"