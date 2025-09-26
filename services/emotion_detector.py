"""
Emotion Detection Service using Hugging Face Inference API (with enhanced model support)
Supports dynamic model selection and improved emotion processing.
"""

import io
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from PIL import Image
import httpx
import asyncio
import structlog

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
            "timestamp": time.time(),
        }


class EmotionDetectorService:
    """Enhanced emotion detection service with dynamic model selection and better processing."""

    def __init__(
        self,
        hf_api_key: str,
        hf_primary: str,
        hf_alternatives: Optional[List[str]] = None,
        groq_api_key: Optional[str] = None,
        openrouter_api_key: Optional[str] = None,
        groq_models: Optional[List[str]] = None,
        openrouter_models: Optional[List[str]] = None,
    ):
        self.hf_api_key = hf_api_key
        self.hf_primary = hf_primary
        self.hf_alternatives = hf_alternatives or []
        self.groq_api_key = groq_api_key
        self.openrouter_api_key = openrouter_api_key
        self.groq_models = groq_models or []
        self.openrouter_models = openrouter_models or []
        self._http = httpx.AsyncClient(timeout=30.0)
        
        # Enhanced emotion mapping for better model compatibility
        self.emotion_mapping = self._get_enhanced_emotion_mapping()
        
        # Model-specific configurations
        self.model_configs = {
            "trpakov/vit-face-expression": {
                "image_size": (224, 224),
                "expected_labels": ["LABEL_0", "LABEL_1", "LABEL_2", "LABEL_3", "LABEL_4", "LABEL_5", "LABEL_6"],
                "label_to_emotion": {
                    "LABEL_0": "angry", "LABEL_1": "disgust", "LABEL_2": "fear",
                    "LABEL_3": "happy", "LABEL_4": "sad", "LABEL_5": "surprise", "LABEL_6": "neutral"
                }
            },
            "dima806/facial_emotions_image_detection": {
                "image_size": (224, 224),
                "expected_labels": ["Angry", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Neutral"],
                "label_to_emotion": {
                    "Angry": "angry", "Happy": "happy", "Sad": "sad",
                    "Surprise": "surprise", "Fear": "fear", "Disgust": "disgust", "Neutral": "neutral"
                }
            },
            "RickyIG/emotion_face_image_classification_v2": {
                "image_size": (224, 224),
                "expected_labels": ["anger", "happy", "sadness", "surprise", "fear", "disgust", "neutral"],
                "label_to_emotion": {
                    "anger": "angry", "happy": "happy", "sadness": "sad",
                    "surprise": "surprise", "fear": "fear", "disgust": "disgust", "neutral": "neutral"
                }
            }
        }
        
        # Track available models (will be populated during health checks)
        self.available_models = []
        self.selected_model = hf_primary

    def _get_enhanced_emotion_mapping(self) -> Dict[str, str]:
        """Enhanced emotion mapping for better model compatibility"""
        return {
            # Standard emotions
            "happy": "happy", "happiness": "happy", "joy": "happy", "smile": "happy",
            "sad": "sad", "sadness": "sad", "sad_face": "sad", "sorrow": "sad",
            "angry": "angry", "anger": "angry", "angry_face": "angry", "mad": "angry",
            "surprise": "surprise", "surprised": "surprise", "surprised_face": "surprise", "shock": "surprise",
            "disgust": "disgust", "disgusted": "disgust", "disgust_face": "disgust", "revulsion": "disgust",
            "fear": "fear", "fearful": "fear", "fear_face": "fear", "scared": "fear", "afraid": "fear",
            "neutral": "neutral", "neutral_face": "neutral", "calm": "neutral", "normal": "neutral",
            
            # Model-specific labels (for different HF models)
            "LABEL_0": "angry", "LABEL_1": "disgust", "LABEL_2": "fear",
            "LABEL_3": "happy", "LABEL_4": "sad", "LABEL_5": "surprise", "LABEL_6": "neutral",
            
            # Alternative spellings and variations
            "contempt": "disgust", "excited": "happy", "confused": "neutral",
            "melancholy": "sad", "furious": "angry", "terrified": "fear",
            "delighted": "happy", "annoyed": "angry", "worried": "fear"
        }

    def _normalize_emotion(self, emotion: Optional[str]) -> str:
        """Normalize emotion label with enhanced mapping"""
        if not emotion:
            return "unknown"
        
        el = emotion.lower().strip()
        normalized = self.emotion_mapping.get(el, el)
        
        # Additional fuzzy matching for unknown labels
        if normalized == el and normalized not in ["happy", "sad", "angry", "surprise", "disgust", "fear", "neutral"]:
            for key, value in self.emotion_mapping.items():
                if key in el or el in key:
                    normalized = value
                    break
        
        return normalized

    def _get_model_config(self, model_id: str) -> dict:
        """Get configuration for specific model"""
        return self.model_configs.get(model_id, {
            "image_size": (224, 224),
            "expected_labels": [],
            "label_to_emotion": {}
        })

    def _preprocess_image(self, image_data: bytes, model_id: str) -> bytes:
        """Enhanced image preprocessing based on model requirements"""
        config = self._get_model_config(model_id)
        image_size = config.get("image_size", (224, 224))
        
        image = Image.open(io.BytesIO(image_data))
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Resize to model-specific requirements
        image = image.resize(image_size, Image.Resampling.LANCZOS)
        
        # Enhanced preprocessing for better results
        # Apply slight contrast enhancement for emotion models
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.1)  # Slight contrast boost
        
        # Save with optimized quality
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=95, optimize=True)
        return buf.getvalue()

    def _process_model_output(self, payload: Any, model_id: str) -> Dict[str, float]:
        """Enhanced processing of model output based on model type"""
        config = self._get_model_config(model_id)
        emotion_scores = {}
        
        if isinstance(payload, list):
            for item in payload:
                raw_label = item.get("label", "unknown")
                score = float(item.get("score", 0.0))
                
                # Use model-specific label mapping if available
                if config.get("label_to_emotion") and raw_label in config["label_to_emotion"]:
                    emotion = config["label_to_emotion"][raw_label]
                else:
                    emotion = self._normalize_emotion(raw_label)
                
                emotion_scores[emotion] = score
                
        elif isinstance(payload, dict):
            # Handle direct dictionary responses
            if "predictions" in payload:
                predictions = payload["predictions"]
                if isinstance(predictions, list) and predictions:
                    return self._process_model_output(predictions[0], model_id)
            elif "scores" in payload:
                scores = payload["scores"]
                labels = payload.get("labels", [])
                for i, score in enumerate(scores):
                    if i < len(labels):
                        raw_label = labels[i]
                        if config.get("label_to_emotion") and raw_label in config["label_to_emotion"]:
                            emotion = config["label_to_emotion"][raw_label]
                        else:
                            emotion = self._normalize_emotion(raw_label)
                        emotion_scores[emotion] = float(score)
        
        # Ensure all basic emotions are present with at least 0 score
        basic_emotions = ["happy", "sad", "angry", "surprise", "disgust", "fear", "neutral"]
        for emotion in basic_emotions:
            if emotion not in emotion_scores:
                emotion_scores[emotion] = 0.0
                
        return emotion_scores

    async def _call_hf_api(self, model_id: str, image_bytes: bytes) -> Optional[Any]:
        """Enhanced HF API call with better error handling"""
        url = f"https://api-inference.huggingface.co/models/{model_id}"
        headers = {"Authorization": f"Bearer {self.hf_api_key}"}
        max_retries = 3
        retry_delay = 1.0

        for attempt in range(max_retries):
            try:
                resp = await self._http.post(
                    url,
                    headers={**headers, "Content-Type": "image/jpeg"},
                    data=image_bytes,
                    timeout=25.0,  # Slightly reduced timeout
                )
            except httpx.TimeoutException:
                logger.warning(f"[HF] Timeout for model {model_id}, attempt {attempt + 1}")
                resp = None
            except Exception as e:
                logger.error(f"[HF] Request error for {model_id}: {e}")
                resp = None

            if resp is None:
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 1.5  # Reduced exponential backoff
                    continue
                return None

            try:
                payload = resp.json()
            except Exception as e:
                logger.error(f"[HF] Invalid JSON from {model_id}: {e}")
                payload = None

            if resp.status_code == 200 and payload is not None:
                logger.info(f"[HF] Success with model {model_id}")
                return payload
            elif resp.status_code == 503:
                est = payload.get("estimated_time", 15) if isinstance(payload, dict) else 15
                logger.info(f"[HF] Model {model_id} loading, wait {est}s")
                await asyncio.sleep(min(est, 20))  # Reduced max wait time
                continue
            elif resp.status_code == 404:
                logger.error(f"[HF] Model {model_id} not found (404)")
                return None
            else:
                logger.error(f"[HF] API error {resp.status_code}: {resp.text}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 1.5
                    continue
                return None

        return None

    async def check_model_availability(self) -> List[str]:
        """Check which models are currently available"""
        available = []
        all_models = [self.hf_primary] + self.hf_alternatives
        
        # Test with a small dummy image
        dummy_img = Image.new("RGB", (100, 100), color="white")
        buf = io.BytesIO()
        dummy_img.save(buf, format="JPEG")
        dummy_bytes = buf.getvalue()
        
        for model in all_models:
            try:
                processed_img = self._preprocess_image(dummy_bytes, model)
                result = await self._call_hf_api(model, processed_img)
                if result is not None:
                    available.append(model)
                    logger.info(f"Model {model} is available")
                else:
                    logger.warning(f"Model {model} is not available")
            except Exception as e:
                logger.error(f"Error checking model {model}: {e}")
        
        self.available_models = available
        return available

    def set_selected_model(self, model_id: str) -> bool:
        """Set the currently selected model"""
        if model_id in ([self.hf_primary] + self.hf_alternatives):
            self.selected_model = model_id
            logger.info(f"Selected model changed to: {model_id}")
            return True
        return False

    async def detect_emotion(
        self,
        image_data: bytes,
        face_bounds: Optional[Dict] = None,
        use_fallback: bool = True,
        selected_model: Optional[str] = None
    ) -> EmotionResult:
        """Enhanced emotion detection with model selection support"""
        start = time.time()
        
        # Use selected model or fallback to current selection
        target_model = selected_model or self.selected_model
        
        try:
            # Enhanced image preprocessing
            processed_image = self._preprocess_image(image_data, target_model)

            # Try selected model first
            payload = await self._call_hf_api(target_model, processed_image)
            used_model = target_model

            # Fallback logic if selected model fails
            if (payload is None or (isinstance(payload, dict) and "error" in payload)) and use_fallback:
                logger.info(f"Primary model {target_model} failed, trying alternatives")
                
                # Try other available models
                fallback_models = [self.hf_primary] + self.hf_alternatives
                fallback_models = [m for m in fallback_models if m != target_model]
                
                for fallback in fallback_models:
                    logger.info(f"[Fallback] Trying model {fallback}")
                    fallback_img = self._preprocess_image(image_data, fallback)
                    fallback_payload = await self._call_hf_api(fallback, fallback_img)
                    if fallback_payload and not (isinstance(fallback_payload, dict) and "error" in fallback_payload):
                        payload = fallback_payload
                        used_model = fallback
                        break

            if payload is None or (isinstance(payload, dict) and "error" in payload):
                error_msg = payload.get("error", "Unknown error") if isinstance(payload, dict) else "No response"
                logger.error(f"All models failed. Last error: {error_msg}")
                return EmotionResult(
                    emotion="unknown",
                    confidence=0.0,
                    all_scores={"error": str(error_msg)},
                    processing_time=time.time() - start,
                    model_used=used_model,
                    face_detected=False
                )

            # Enhanced output processing
            emotion_scores = self._process_model_output(payload, used_model)
            
            if not emotion_scores:
                logger.warning(f"No valid emotion scores from model {used_model}")
                return EmotionResult(
                    emotion="neutral",
                    confidence=0.5,
                    all_scores={"neutral": 0.5},
                    processing_time=time.time() - start,
                    model_used=used_model,
                    face_detected=True
                )

            # Get top emotion
            top_emotion = max(emotion_scores, key=emotion_scores.get)
            top_confidence = emotion_scores[top_emotion]
            
            # Ensure confidence is reasonable (some models return very low scores)
            if top_confidence < 0.3:
                # Apply confidence boosting for low-confidence predictions
                boost_factor = 1.3
                emotion_scores = {k: min(v * boost_factor, 1.0) for k, v in emotion_scores.items()}
                top_confidence = emotion_scores[top_emotion]

            logger.info(f"Detected emotion: {top_emotion} ({top_confidence:.3f}) using {used_model}")
            
            return EmotionResult(
                emotion=top_emotion,
                confidence=top_confidence,
                all_scores=emotion_scores,
                processing_time=time.time() - start,
                model_used=used_model,
                face_detected=True,
                landmarks=face_bounds,
            )

        except Exception as e:
            logger.error(f"Emotion detection exception: {e}", exc_info=True)
            return EmotionResult(
                emotion="error",
                confidence=0.0,
                all_scores={"error": str(e)},
                processing_time=time.time() - start,
                model_used=target_model,
                face_detected=False,
            )

    async def batch_detect(
        self,
        image_list: List[bytes],
        selected_model: Optional[str] = None
    ) -> List[EmotionResult]:
        """Enhanced batch processing with model selection"""
        tasks = [
            self.detect_emotion(img, selected_model=selected_model)
            for img in image_list
        ]
        return await asyncio.gather(*tasks, return_exceptions=False)

    async def warmup(self):
        """Enhanced warmup with model availability check"""
        logger.info("Warming up emotion detector...")
        
        # Check model availability
        await self.check_model_availability()
        
        # Warm up available models
        dummy_img = Image.new("RGB", (224, 224), color="white")
        buf = io.BytesIO()
        dummy_img.save(buf, format="JPEG")
        dummy_bytes = buf.getvalue()
        
        for model in self.available_models[:2]:  # Warm up only first 2 to save time
            try:
                logger.info(f"Warming up model: {model}")
                await self.detect_emotion(dummy_bytes, selected_model=model, use_fallback=False)
            except Exception as e:
                logger.warning(f"Warmup failed for {model}: {e}")
        
        logger.info(f"Warmup complete. Available models: {len(self.available_models)}")

    async def close(self):
        """Clean up resources"""
        await self._http.aclose()

    # Removed unnecessary Groq/OpenRouter stubs to save memory and reduce code complexity
    # Original code commented out:
    # async def _call_groq_api(self, model_id: str, image_bytes: bytes) -> Optional[Any]:
    #     logger.debug(f"[Groq] Stub call for model {model_id}")
    #     return None
    # 
    # async def _call_openrouter_api(self, model_id: str, image_bytes: bytes) -> Optional[Any]:
    #     logger.debug(f"[OpenRouter] Stub call for model {model_id}")
    #     return None


# Singleton instance management
_detector_instance: Optional["EmotionDetectorService"] = None


async def get_emotion_detector(settings) -> "EmotionDetectorService":
    """Get or create emotion detector instance"""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = EmotionDetectorService(
            hf_api_key=settings.hf_api_key,
            hf_primary=settings.hf_primary_model,
            hf_alternatives=settings.hf_alternative_models,
            # Removed Groq/OpenRouter initialization to simplify and save memory
            # groq_models=settings.groq_model_list,
            # openrouter_models=settings.openrouter_model_list
        )
        await _detector_instance.warmup()
    return _detector_instance