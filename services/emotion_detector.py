"""
Emotion Detection Service using Hugging Face Inference API
Optimized for low memory usage and high performance
"""

import base64
import io
import time
from typing import Dict, List, Optional, Tuple, Any
from PIL import Image
import numpy as np
import httpx
import asyncio
from dataclasses import dataclass
import json
import structlog
from functools import lru_cache

logger = structlog.get_logger()


@dataclass
class EmotionResult:
    """Emotion detection result with confidence scores"""
    emotion: str
    confidence: float
    all_scores: Dict[str, float]
    processing_time: float
    model_used: str
    face_detected: bool = True
    landmarks: Optional[Dict] = None
    
    def to_dict(self) -> dict:
        return {
            "emotion": self.emotion,
            "confidence": self.confidence,
            "all_scores": self.all_scores,
            "processing_time": self.processing_time,
            "model_used": self.model_used,
            "face_detected": self.face_detected,
            "landmarks": self.landmarks,
            "timestamp": time.time()
        }


class EmotionDetectorService:
    """
    Production-ready emotion detection service using Hugging Face API
    """
    
    def __init__(self, api_key: str, model_id: str, alternative_models: List[str] = None):
        self.api_key = api_key
        self.model_id = model_id
        self.alternative_models = alternative_models or []
        self.headers = {"Authorization": f"Bearer {api_key}"}
        self.client = httpx.AsyncClient(timeout=30.0)
        self._model_status_cache = {}
        self.emotion_mapping = self._get_emotion_mapping()
        
    def _get_emotion_mapping(self) -> Dict[str, str]:
        """Standardize emotion labels across different models"""
        return {
            # Standard mappings
            "happy": "happy",
            "happiness": "happy",
            "joy": "happy",
            "sad": "sad",
            "sadness": "sad",
            "angry": "angry",
            "anger": "angry",
            "surprise": "surprise",
            "surprised": "surprise",
            "disgust": "disgust",
            "disgusted": "disgust",
            "fear": "fear",
            "fearful": "fear",
            "neutral": "neutral",
            "contempt": "disgust",
            # Additional mappings
            "excited": "happy",
            "calm": "neutral",
            "confused": "neutral",
        }
    
    def _normalize_emotion(self, emotion: str) -> str:
        """Normalize emotion label to standard format"""
        emotion_lower = emotion.lower().strip()
        return self.emotion_mapping.get(emotion_lower, emotion_lower)
    
    async def _check_model_status(self, model_id: str) -> bool:
        """Check if model is loaded and ready"""
        try:
            # Cache status for 5 minutes
            cache_key = f"status_{model_id}"
            if cache_key in self._model_status_cache:
                cached_time, status = self._model_status_cache[cache_key]
                if time.time() - cached_time < 300:
                    return status
            
            url = f"https://api-inference.huggingface.co/models/{model_id}"
            response = await self.client.get(url, headers=self.headers)
            status = response.status_code == 200
            
            self._model_status_cache[cache_key] = (time.time(), status)
            return status
        except Exception as e:
            logger.warning(f"Failed to check model status: {e}")
            return False
    
    def _preprocess_image(self, image: Image.Image, target_size: Tuple[int, int] = (224, 224)) -> Image.Image:
        """
        Preprocess image for optimal inference
        - Resize to target size
        - Convert to RGB
        - Apply basic normalization
        """
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Smart crop to face region if detected (placeholder for client-side data)
        # In production, face bounds come from client-side MediaPipe
        
        # Resize with high-quality resampling
        image = image.resize(target_size, Image.Resampling.LANCZOS)
        
        return image
    
    async def _call_inference_api(self, image_bytes: bytes, model_id: str) -> List[Dict]:
        """Call Hugging Face inference API with retry logic"""
        url = f"https://api-inference.huggingface.co/models/{model_id}"
        
        max_retries = 3
        retry_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                response = await self.client.post(
                    url,
                    headers=self.headers,
                    data=image_bytes,
                    timeout=20.0
                )
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 503:
                    # Model is loading
                    error_data = response.json()
                    estimated_time = error_data.get("estimated_time", 20)
                    logger.info(f"Model loading, waiting {estimated_time}s...")
                    await asyncio.sleep(min(estimated_time, 30))
                    continue
                else:
                    logger.error(f"API error: {response.status_code} - {response.text}")
                    
            except httpx.TimeoutException:
                logger.warning(f"Timeout on attempt {attempt + 1}")
            except Exception as e:
                logger.error(f"Inference error: {e}")
            
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)
                retry_delay *= 2
        
        return []
    
    async def detect_emotion(
        self, 
        image_data: bytes,
        face_bounds: Optional[Dict] = None,
        use_fallback: bool = True
    ) -> EmotionResult:
        """
        Detect emotion from image bytes
        
        Args:
            image_data: Image bytes (JPEG/PNG)
            face_bounds: Optional face bounding box from client
            use_fallback: Try alternative models if primary fails
        
        Returns:
            EmotionResult with detected emotion and confidence
        """
        start_time = time.time()
        
        try:
            # Load and preprocess image
            image = Image.open(io.BytesIO(image_data))
            processed_image = self._preprocess_image(image)
            
            # Convert to bytes for API
            img_buffer = io.BytesIO()
            processed_image.save(img_buffer, format='JPEG', quality=95)
            img_bytes = img_buffer.getvalue()
            
            # Try primary model
            results = await self._call_inference_api(img_bytes, self.model_id)
            model_used = self.model_id
            
            # Fallback to alternative models if needed
            if not results and use_fallback:
                for alt_model in self.alternative_models:
                    logger.info(f"Trying fallback model: {alt_model}")
                    results = await self._call_inference_api(img_bytes, alt_model)
                    if results:
                        model_used = alt_model
                        break
            
            if not results:
                return EmotionResult(
                    emotion="unknown",
                    confidence=0.0,
                    all_scores={},
                    processing_time=time.time() - start_time,
                    model_used="none",
                    face_detected=False
                )
            
            # Process results
            emotion_scores = {}
            for item in results:
                label = self._normalize_emotion(item.get("label", "unknown"))
                score = item.get("score", 0.0)
                emotion_scores[label] = score
            
            # Get top emotion
            if emotion_scores:
                top_emotion = max(emotion_scores, key=emotion_scores.get)
                confidence = emotion_scores[top_emotion]
            else:
                top_emotion = "neutral"
                confidence = 0.5
            
            return EmotionResult(
                emotion=top_emotion,
                confidence=confidence,
                all_scores=emotion_scores,
                processing_time=time.time() - start_time,
                model_used=model_used,
                face_detected=True,
                landmarks=face_bounds
            )
            
        except Exception as e:
            logger.error(f"Emotion detection failed: {e}")
            return EmotionResult(
                emotion="error",
                confidence=0.0,
                all_scores={},
                processing_time=time.time() - start_time,
                model_used="error",
                face_detected=False
            )
    
    async def batch_detect(self, images: List[bytes], max_concurrent: int = 5) -> List[EmotionResult]:
        """Process multiple images concurrently with rate limiting"""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_with_limit(img_data):
            async with semaphore:
                return await self.detect_emotion(img_data)
        
        tasks = [process_with_limit(img) for img in images]
        return await asyncio.gather(*tasks)
    
    async def warmup(self):
        """Warmup the model by sending a test request"""
        try:
            # Create a small test image
            test_image = Image.new('RGB', (224, 224), color='white')
            img_buffer = io.BytesIO()
            test_image.save(img_buffer, format='JPEG')
            
            logger.info(f"Warming up model: {self.model_id}")
            result = await self.detect_emotion(img_buffer.getvalue(), use_fallback=False)
            logger.info(f"Warmup complete: {result.emotion}")
            return True
        except Exception as e:
            logger.error(f"Warmup failed: {e}")
            return False
    
    async def close(self):
        """Cleanup resources"""
        await self.client.aclose()


# Singleton instance management
_detector_instance: Optional[EmotionDetectorService] = None


async def get_emotion_detector(settings) -> EmotionDetectorService:
    """Get or create emotion detector instance"""
    global _detector_instance
    
    if _detector_instance is None:
        _detector_instance = EmotionDetectorService(
            api_key=settings.hf_api_key,
            model_id=settings.hf_model_id,
            alternative_models=settings.alternative_models
        )
        await _detector_instance.warmup()
    
    return _detector_instance