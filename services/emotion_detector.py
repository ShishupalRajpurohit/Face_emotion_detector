"""
Emotion Detection Service using Hugging Face Inference API (with fallback logic)
Supports primary + alternative HF models and placeholders for Groq / OpenRouter.
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
    """Emotion detection service with fallback support across HF models."""

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
        self.emotion_mapping = self._get_emotion_mapping()

    def _get_emotion_mapping(self) -> Dict[str, str]:
        return {
            # General emotions
            "happy": "happy", "happiness": "happy", "joy": "happy", "smile": "happy",
            "sad": "sad", "sadness": "sad", "sad_face": "sad",
            "angry": "angry", "anger": "angry", "angry_face": "angry",
            "surprise": "surprise", "surprised": "surprise", "surprised_face": "surprise",
            "disgust": "disgust", "disgusted": "disgust", "disgust_face": "disgust",
            "fear": "fear", "fearful": "fear", "fear_face": "fear",
            "neutral": "neutral", "neutral_face": "neutral", "calm": "neutral",
            # Optional extensions
            "contempt": "disgust", "excited": "happy", "confused": "neutral",
        }


    def _normalize_emotion(self, emotion: Optional[str]) -> str:
        if not emotion:
            return "unknown"
        el = emotion.lower().strip()
        return self.emotion_mapping.get(el, el)

    async def _call_hf_api(self, model_id: str, image_bytes: bytes) -> Optional[Any]:
        """Call Hugging Face inference API; return JSON payload or None."""
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
                    timeout=20.0,
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
                    retry_delay *= 2
                    continue
                return None

            try:
                payload = resp.json()
            except Exception as e:
                logger.error(f"[HF] Invalid JSON from {model_id}: {e}, text={resp.text}")
                payload = None

            if resp.status_code == 200 and payload is not None:
                return payload
            elif resp.status_code == 503:
                est = payload.get("estimated_time", 20) if isinstance(payload, dict) else 20
                logger.info(f"[HF] Model {model_id} loading, wait {est}s (attempt {attempt+1})")
                await asyncio.sleep(min(est, 30))
                continue
            elif resp.status_code == 404:
                logger.error(f"[HF] Model {model_id} not found (404)")
                return None
            else:
                logger.error(f"[HF] API error {resp.status_code}: {resp.text}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2
                    continue
                return None

        return None

    async def detect_emotion(
        self,
        image_data: bytes,
        face_bounds: Optional[Dict] = None,
        use_fallback: bool = True
    ) -> EmotionResult:
        start = time.time()
        try:
            image = Image.open(io.BytesIO(image_data))
            if image.mode != "RGB":
                image = image.convert("RGB")
            image = image.resize((224, 224), Image.Resampling.LANCZOS)
            buf = io.BytesIO()
            image.save(buf, format="JPEG", quality=95)
            img_bytes = buf.getvalue()

            # 1. Try HF primary
            payload = await self._call_hf_api(self.hf_primary, img_bytes)
            used = self.hf_primary

            # 2. Fallback to alternative HF models
            if (payload is None) or (isinstance(payload, dict) and "error" in payload):
                for alt in self.hf_alternatives:
                    logger.info(f"[Fallback-HF] trying alternative model {alt}")
                    alt_payload = await self._call_hf_api(alt, img_bytes)
                    if alt_payload:
                        payload = alt_payload
                        used = alt
                        break

            # 3. If all HF failed, try Groq
            if payload is None and use_fallback:
                for gm in self.groq_models:
                    logger.info(f"[Fallback-Groq] trying model {gm}")
                    gp = await self._call_groq_api(gm, img_bytes)
                    if gp:
                        payload = gp
                        used = gm
                        break

            # 4. Try OpenRouter
            if payload is None and use_fallback:
                for om in self.openrouter_models:
                    logger.info(f"[Fallback-OR] trying model {om}")
                    op = await self._call_openrouter_api(om, img_bytes)
                    if op:
                        payload = op
                        used = om
                        break

            if payload is None:
                return EmotionResult(
                    emotion="unknown",
                    confidence=0.0,
                    all_scores={},
                    processing_time=time.time() - start,
                    model_used=used,
                    face_detected=False
                )

            # Normalize HF payload (list of dicts with label/score)
            emotion_scores: Dict[str, float] = {}
            if isinstance(payload, list):
                for item in payload:
                    raw_label = item.get("label") or "unknown"
                    score = float(item.get("score", 0.0) or 0.0)
                    emotion_scores[self._normalize_emotion(raw_label)] = score
            elif isinstance(payload, dict) and "error" in payload:
                logger.error(f"Payload error from {used}: {payload}")
                return EmotionResult(
                    emotion="error",
                    confidence=0.0,
                    all_scores={"error": str(payload.get("error"))},
                    processing_time=time.time() - start,
                    model_used=used,
                    face_detected=False,
                )

            top_emotion = max(emotion_scores, key=emotion_scores.get) if emotion_scores else "neutral"
            top_confidence = emotion_scores.get(top_emotion, 0.5)

            return EmotionResult(
                emotion=top_emotion,
                confidence=top_confidence,
                all_scores=emotion_scores,
                processing_time=time.time() - start,
                model_used=used,
                face_detected=True,
                landmarks=face_bounds,
            )

        except Exception as e:
            logger.error(f"Emotion detection exception: {e}")
            return EmotionResult(
                emotion="error",
                confidence=0.0,
                all_scores={},
                processing_time=time.time() - start,
                model_used="error",
                face_detected=False,
            )

    async def warmup(self):
        """Warmup by sending a dummy image"""
        white = Image.new("RGB", (224, 224), color="white")
        buf = io.BytesIO()
        white.save(buf, format="JPEG")
        await self.detect_emotion(buf.getvalue(), use_fallback=False)

    async def close(self):
        await self._http.aclose()

    async def _call_groq_api(self, model_id: str, image_bytes: bytes) -> Optional[Any]:
        logger.debug(f"[Groq] Stub call for model {model_id}")
        return None

    async def _call_openrouter_api(self, model_id: str, image_bytes: bytes) -> Optional[Any]:
        logger.debug(f"[OpenRouter] Stub call for model {model_id}")
        return None


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
            groq_models=settings.groq_model_list,
            openrouter_models=settings.openrouter_model_list
        )
        await _detector_instance.warmup()
    return _detector_instance
