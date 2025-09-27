"""
Enhanced application configuration with multi-provider support
Includes Hugging Face, Groq, and OpenRouter API configurations
"""

from typing import List, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
import secrets


class Settings(BaseSettings):
    """Application settings with multi-provider environment variable support"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # === Hugging Face Configuration ===
    hf_api_key: str
    hf_primary_model: str = "trpakov/vit-face-expression"
    hf_alt_models: Optional[str] = "dima806/facial_emotions_image_detection,RickyIG/emotion_face_image_classification_v2"

    # === Groq API Configuration ===
    groq_api_key: Optional[str] = None
    groq_models: Optional[str] = "llava-v1.5-7b-4096-preview,llama-3.2-11b-vision-preview"

    # === OpenRouter API Configuration ===
    openrouter_api_key: Optional[str] = None
    openrouter_models: Optional[str] = "liuhaotian/llava-13b,google/gemini-pro-vision"

    # === Server Configuration ===
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    log_level: str = "info"

    # === Security ===
    secret_key: str = secrets.token_urlsafe(32)
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30

    # === Rate Limiting ===
    rate_limit_requests: int = 100
    rate_limit_period: int = 60

    # === Cache Configuration ===
    cache_ttl: int = 300
    enable_cache: bool = True

    # === CORS Settings ===
    cors_origins: List[str] = ["*"]
    cors_credentials: bool = True
    cors_methods: List[str] = ["*"]
    cors_headers: List[str] = ["*"]

    # === Environment ===
    environment: str = "development"
    debug: bool = False

    # === Performance Settings ===
    max_image_size: int = 2 * 1024 * 1024  # Increased to 2MB for better quality
    max_batch_size: int = 5  # Reduced for better performance
    request_timeout: int = 45  # Increased timeout for multimodal models

    # === Enhanced Emotion Labels Mapping ===
    emotion_labels: dict = {
        "happy": {"emoji": "😊", "color": "#4CAF50", "description": "Joy, happiness, contentment"},
        "sad": {"emoji": "😢", "color": "#2196F3", "description": "Sadness, sorrow, melancholy"},
        "angry": {"emoji": "😠", "color": "#F44336", "description": "Anger, frustration, rage"},
        "surprise": {"emoji": "😮", "color": "#FF9800", "description": "Surprise, shock, amazement"},
        "disgust": {"emoji": "🤢", "color": "#9C27B0", "description": "Disgust, revulsion, dislike"},
        "fear": {"emoji": "😨", "color": "#795548", "description": "Fear, anxiety, worry"},
        "neutral": {"emoji": "😐", "color": "#9E9E9E", "description": "Neutral, calm, no strong emotion"},
    }

    @property
    def available_models(self) -> List[dict]:
        """Get comprehensive list of available models with enhanced metadata"""
        models = []
        
        # Hugging Face Models
        hf_model_info = {
            "trpakov/vit-face-expression": {
                "name": "ViT Face Expression",
                "full_name": "trpakov/vit-face-expression",
                "company": "trpakov",
                "provider": "Hugging Face",
                "description": "Vision Transformer trained on FER2013 dataset with 7 emotion classes",
                "accuracy": "85%",
                "speed": "Fast (~200ms)",
                "model_type": "Vision Transformer",
                "training_data": "FER2013 Dataset",
                "strengths": "Fast inference, good general accuracy"
            },
            "dima806/facial_emotions_image_detection": {
                "name": "Facial Emotions Detection",
                "full_name": "dima806/facial_emotions_image_detection", 
                "company": "dima806",
                "provider": "Hugging Face",
                "description": "High accuracy emotion detection model with optimized preprocessing",
                "accuracy": "91%",
                "speed": "Medium (~400ms)",
                "model_type": "CNN-based",
                "training_data": "Enhanced emotion datasets",
                "strengths": "High accuracy, robust to lighting"
            },
            "RickyIG/emotion_face_image_classification_v2": {
                "name": "Emotion Classification v2",
                "full_name": "RickyIG/emotion_face_image_classification_v2",
                "company": "RickyIG", 
                "provider": "Hugging Face",
                "description": "Optimized emotion classification model with improved architecture",
                "accuracy": "80%",
                "speed": "Fast (~250ms)",
                "model_type": "Optimized CNN",
                "training_data": "Multiple emotion datasets",
                "strengths": "Good balance of speed and accuracy"
            }
        }
        
        # Add primary HF model
        if self.hf_primary_model in hf_model_info:
            model_data = hf_model_info[self.hf_primary_model].copy()
            model_data["id"] = self.hf_primary_model
            model_data["is_primary"] = True
            models.append(model_data)
        
        # Add alternative HF models
        for alt_model in self.hf_alternative_models:
            if alt_model in hf_model_info:
                model_data = hf_model_info[alt_model].copy()
                model_data["id"] = alt_model
                model_data["is_primary"] = False
                models.append(model_data)
        
        # Groq Models (if API key available)
        if self.groq_api_key:
            groq_model_info = {
                "llava-v1.5-7b-4096-preview": {
                    "name": "LLaVA 1.5 7B",
                    "full_name": "llava-v1.5-7b-4096-preview",
                    "company": "Microsoft/LLaVA Team",
                    "provider": "Groq",
                    "description": "Large multimodal model for visual understanding and emotion detection",
                    "accuracy": "75%",
                    "speed": "Very Fast (~150ms)",
                    "model_type": "Large Language + Vision Model",
                    "training_data": "Visual instruction tuning dataset",
                    "strengths": "Contextual understanding, natural language reasoning"
                },
                "llama-3.2-11b-vision-preview": {
                    "name": "Llama 3.2 11B Vision",
                    "full_name": "llama-3.2-11b-vision-preview",
                    "company": "Meta",
                    "provider": "Groq",
                    "description": "Meta's latest multimodal model with vision capabilities",
                    "accuracy": "80%",
                    "speed": "Fast (~200ms)",
                    "model_type": "Transformer + Vision Encoder", 
                    "training_data": "Multimodal web data",
                    "strengths": "Latest architecture, good reasoning"
                }
            }
            
            for groq_model in self.groq_model_list:
                if groq_model in groq_model_info:
                    model_data = groq_model_info[groq_model].copy()
                    model_data["id"] = groq_model
                    model_data["is_primary"] = False
                    models.append(model_data)
        
        # OpenRouter Models (if API key available)
        if self.openrouter_api_key:
            openrouter_model_info = {
                "liuhaotian/llava-13b": {
                    "name": "LLaVA 13B",
                    "full_name": "liuhaotian/llava-13b",
                    "company": "LLaVA Team",
                    "provider": "OpenRouter",
                    "description": "Large multimodal model for complex visual reasoning",
                    "accuracy": "78%",
                    "speed": "Medium (~500ms)",
                    "model_type": "Large Multimodal Model",
                    "training_data": "Visual instruction following dataset",
                    "strengths": "Complex reasoning, detailed analysis"
                },
                "google/gemini-pro-vision": {
                    "name": "Gemini Pro Vision",
                    "full_name": "google/gemini-pro-vision",
                    "company": "Google",
                    "provider": "OpenRouter", 
                    "description": "Google's powerful multimodal AI for vision and language tasks",
                    "accuracy": "85%",
                    "speed": "Medium (~600ms)",
                    "model_type": "Multimodal Transformer",
                    "training_data": "Large-scale multimodal data",
                    "strengths": "High accuracy, robust performance"
                }
            }
            
            for or_model in self.openrouter_model_list:
                if or_model in openrouter_model_info:
                    model_data = openrouter_model_info[or_model].copy()
                    model_data["id"] = or_model
                    model_data["is_primary"] = False
                    models.append(model_data)
        
        return models

    @property
    def hf_inference_url(self) -> str:
        """Generate Hugging Face inference API URL"""
        return f"https://api-inference.huggingface.co/models/{self.hf_primary_model}"

    @property
    def hf_alternative_models(self) -> List[str]:
        """Parse HF_ALT_MODELS env var into list"""
        if not self.hf_alt_models:
            return []
        return [m.strip() for m in self.hf_alt_models.split(",") if m.strip()]

    @property
    def groq_model_list(self) -> List[str]:
        """Parse GROQ_MODELS env var into list"""
        if not self.groq_models:
            return []
        return [m.strip() for m in self.groq_models.split(",") if m.strip()]

    @property
    def openrouter_model_list(self) -> List[str]:
        """Parse OPENROUTER_MODELS env var into list"""
        if not self.openrouter_models:
            return []
        return [m.strip() for m in self.openrouter_models.split(",") if m.strip()]

    def get_provider_info(self) -> dict:
        """Get information about available providers"""
        providers = {
            "huggingface": {
                "name": "Hugging Face",
                "status": "available" if self.hf_api_key else "missing_key",
                "models_count": 1 + len(self.hf_alternative_models),
                "description": "Specialized emotion detection models"
            }
        }
        
        if self.groq_api_key:
            providers["groq"] = {
                "name": "Groq",
                "status": "available",
                "models_count": len(self.groq_model_list),
                "description": "Fast multimodal inference"
            }
        else:
            providers["groq"] = {
                "name": "Groq", 
                "status": "missing_key",
                "models_count": 0,
                "description": "Fast multimodal inference (requires API key)"
            }
        
        if self.openrouter_api_key:
            providers["openrouter"] = {
                "name": "OpenRouter",
                "status": "available", 
                "models_count": len(self.openrouter_model_list),
                "description": "Access to multiple AI providers"
            }
        else:
            providers["openrouter"] = {
                "name": "OpenRouter",
                "status": "missing_key",
                "models_count": 0,
                "description": "Access to multiple AI providers (requires API key)"
            }
        
        return providers

    def validate_settings(self) -> None:
        """Enhanced settings validation with multi-provider support"""
        if not self.hf_api_key or self.hf_api_key == "hf_xxxxxxxxxxxxxxxxxxxxx":
            raise ValueError(
                "HF_API_KEY is not set! Get your API key from: "
                "https://huggingface.co/settings/tokens"
            )

        # Validate optional API keys format
        if self.groq_api_key and not self.groq_api_key.startswith("gsk_"):
            raise ValueError("Invalid GROQ_API_KEY format. Should start with 'gsk_'")
        
        if self.openrouter_api_key and not self.openrouter_api_key.startswith("sk-or-"):
            raise ValueError("Invalid OPENROUTER_API_KEY format. Should start with 'sk-or-'")

        if self.environment == "production":
            if self.debug:
                raise ValueError("Debug mode should be disabled in production")
            if len(self.secret_key) < 32:
                raise ValueError("Please set a strong SECRET_KEY for production (32+ characters)")


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    settings = Settings()
    settings.validate_settings()
    return settings