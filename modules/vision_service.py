"""
Vision model service for communicating with LM Studio
"""

import base64
from pathlib import Path
from typing import Optional

from openai import OpenAI

from .config import Config
from .models import ImageDescription


class VisionModelService:
    """Handles communication with LM Studio vision model"""
    
    def __init__(self, base_url: str = Config.LM_STUDIO_URL, model: str = Config.MODEL_NAME):
        self.client = OpenAI(base_url=base_url, api_key="not-needed")
        self.model = model
    
    def _get_json_schema(self) -> dict:
        """Get JSON schema for structured output"""
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "image_description",
                "schema": ImageDescription.model_json_schema()
            }
        }
    
    def _encode_image(self, image_path: Path) -> str:
        """Encode image to base64"""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    
    def describe_image(self, image_path: Path) -> Optional[ImageDescription]:
        """
        Get structured description of an image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            ImageDescription object or None if processing fails
        """
        try:
            image_b64 = self._encode_image(image_path)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "What's on this image? Return JSON with two fields: "
                                "`description` (a short description), and "
                                "`name` (a few words with no spaces)."
                            )
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_b64}"
                            }
                        }
                    ]
                }],
                response_format=self._get_json_schema()
            )
            
            raw_response = response.choices[0].message.content
            return ImageDescription.model_validate_json(raw_response)
            
        except Exception as e:
            print(f"Error describing image {image_path}: {e}")
            return None