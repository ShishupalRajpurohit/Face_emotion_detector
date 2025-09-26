"""
Application configuration using Pydantic Settings for type safety and validation
"""

from typing import List, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
import secrets


class Settings(BaseSettings):
    """Application settings with environment variable support"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # ignore unknown fields in .env
    )

    # === Hugging Face Configuration ===
    hf_api_key: str
    hf_primary_model: str = "trpakov/vit-face-expression"
    hf_alt_models: Optional[str] = None  # comma-separated in .env

    # === Alternative API Keys (optional) ===
    groq_api_key: Optional[str] = None
    openrouter_api_key: Optional[str] = None

    # === Fallback Models for Groq / OpenRouter ===
    groq_models: Optional[str] = None  # comma-separated in .env
    openrouter_models: Optional[str] = None

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

    # === Monitoring ===
    enable_metrics: bool = True
    # metrics_port: int = 9090  # Removed - not needed for deployment, saves memory

    # === Environment ===
    environment: str = "development"
    debug: bool = False

    # === Performance Settings ===
    max_image_size: int = 1024 * 1024  # 1MB
    max_batch_size: int = 10
    request_timeout: int = 30

    # === Emotion Labels Mapping (updated for consistency) ===
    emotion_labels: dict = {
        "happy": {"emoji": "😊", "color": "#4CAF50"},
        "sad": {"emoji": "😢", "color": "#2196F3"},
        "angry": {"emoji": "😠", "color": "#F44336"},
        "surprise": {"emoji": "😮", "color": "#FF9800"},
        "disgust": {"emoji": "🤢", "color": "#9C27B0"},
        "fear": {"emoji": "😨", "color": "#795548"},
        "neutral": {"emoji": "😐", "color": "#9E9E9E"},
    }

    # === Model Configuration (NEW - for model selection) ===
    @property
    def available_models(self) -> List[dict]:
        """Get list of available models with metadata"""
        models = [
            {
                "id": self.hf_primary_model,
                "name": "ViT Face Expression (Primary)",
                "provider": "huggingface",
                "description": "Vision Transformer trained on FER2013 dataset",
                "accuracy": "~85%",
                "speed": "Fast"
            }
        ]
        
        # Add alternative HF models
        for alt_model in self.hf_alternative_models:
            if alt_model == "dima806/facial_emotions_image_detection":
                models.append({
                    "id": alt_model,
                    "name": "Facial Emotions Detection",
                    "provider": "huggingface", 
                    "description": "High accuracy emotion detection model",
                    "accuracy": "~91%",
                    "speed": "Medium"
                })
            elif alt_model == "RickyIG/emotion_face_image_classification_v2":
                models.append({
                    "id": alt_model,
                    "name": "Emotion Classification v2",
                    "provider": "huggingface",
                    "description": "Optimized emotion classification",
                    "accuracy": "~80%", 
                    "speed": "Fast"
                })
            else:
                models.append({
                    "id": alt_model,
                    "name": alt_model.split("/")[-1].replace("_", " ").title(),
                    "provider": "huggingface",
                    "description": "Alternative emotion detection model",
                    "accuracy": "Unknown",
                    "speed": "Medium"
                })
        
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
        if not self.groq_models:
            return []
        return [m.strip() for m in self.groq_models.split(",") if m.strip()]

    @property
    def openrouter_model_list(self) -> List[str]:
        if not self.openrouter_models:
            return []
        return [m.strip() for m in self.openrouter_models.split(",") if m.strip()]

    def validate_settings(self) -> None:
        """Validate critical settings"""
        if not self.hf_api_key or self.hf_api_key == "hf_xxxxxxxxxxxxxxxxxxxxx":
            raise ValueError(
                "HF_API_KEY is not set! Get your API key from: "
                "https://huggingface.co/settings/tokens"
            )

        if self.environment == "production":
            if self.debug:
                raise ValueError("Debug mode should be disabled in production")
            if self.secret_key == secrets.token_urlsafe(32):
                raise ValueError("Please set a custom SECRET_KEY for production")


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    settings = Settings()
    settings.validate_settings()
    return settings