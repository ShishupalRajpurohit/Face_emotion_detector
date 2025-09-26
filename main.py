"""
Production-ready FastAPI server for Facial Emotion Detection
Optimized for Render free tier (< 256MB RAM)
"""

import os
import time
import json
import base64
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager
import asyncio
from datetime import datetime, timedelta
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File, WebSocket, WebSocketDisconnect, Depends, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, validator
import uvicorn
import structlog
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from config import get_settings

# Import the service after config to ensure proper initialization
try:
    from services.emotion_detector import get_emotion_detector, EmotionResult, EmotionDetectorService
except ImportError as e:
    print(f"Error importing emotion detector: {e}")
    # We'll handle this in the lifespan function

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.dev.ConsoleRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Load settings
settings = get_settings()

# Rate limiting
limiter = Limiter(key_func=get_remote_address)


# Request/Response Models
class EmotionRequest(BaseModel):
    """Request model for emotion detection"""
    image: str  # Base64 encoded image
    face_bounds: Optional[Dict[str, float]] = None
    return_all_scores: bool = True
    
    @validator('image')
    def validate_image(cls, v):
        try:
            # Check if it's valid base64
            if ',' in v:  # Handle data URL format
                v = v.split(',')[1]
            base64.b64decode(v)
            return v
        except Exception:
            raise ValueError("Invalid base64 image data")


class EmotionResponse(BaseModel):
    """Response model for emotion detection"""
    success: bool
    emotion: str
    confidence: float
    all_scores: Optional[Dict[str, float]] = None
    processing_time: float
    model_used: str
    face_detected: bool
    landmarks: Optional[Dict] = None
    timestamp: float


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str = "1.0.0"
    uptime: float
    memory_usage_mb: float
    environment: str


class WebSocketMessage(BaseModel):
    """WebSocket message format"""
    type: str  # 'frame', 'config', 'error'
    data: Any


# Application lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    # Startup
    logger.info("Starting Facial Emotion Detector API...")
    app.state.start_time = time.time()
    
    # Initialize emotion detector
    try:
        from services.emotion_detector import get_emotion_detector
        detector = await get_emotion_detector(settings)
        app.state.detector = detector
        logger.info("Emotion detector initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize detector: {e}")
        # Don't raise - let the app start anyway for health checks
        app.state.detector = None
    
    # Initialize WebSocket connections manager
    app.state.connections = set()
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")
    if hasattr(app.state, 'detector') and app.state.detector:
        await app.state.detector.close()
    
    # Close all WebSocket connections
    for websocket in app.state.connections.copy():
        try:
            await websocket.close()
        except:
            pass


# Create FastAPI app
app = FastAPI(
    title="Facial Emotion Detector API",
    description="Production-ready emotion detection service optimized for low resource usage",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/api/docs" if settings.debug else None,
    redoc_url="/api/redoc" if settings.debug else None,
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=settings.cors_credentials,
    allow_methods=settings.cors_methods,
    allow_headers=settings.cors_headers,
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Add rate limit handler
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# Dependency injection
async def get_detector():
    """Get emotion detector instance"""
    if not hasattr(app.state, 'detector') or app.state.detector is None:
        raise HTTPException(
            status_code=503, 
            detail="Emotion detector not available. Please check server logs."
        )
    return app.state.detector


# Health & Monitoring Endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for monitoring"""
    try:
        import psutil
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
    except ImportError:
        memory_mb = 0.0
    
    return HealthResponse(
        status="healthy",
        uptime=time.time() - app.state.start_time,
        memory_usage_mb=round(memory_mb, 2),
        environment=settings.environment
    )


@app.get("/api/status")
async def api_status():
    """Get API status and configuration"""
    detector_status = "available" if hasattr(app.state, 'detector') and app.state.detector else "unavailable"
    
    return {
        "status": "operational",
        "detector_status": detector_status,
        "model": settings.hf_model_id,
        "available_emotions": list(settings.emotion_labels.keys()),
        "max_image_size": settings.max_image_size,
        "rate_limits": {
            "requests_per_minute": settings.rate_limit_requests,
        }
    }


# Main API Endpoints
@app.post("/api/detect", response_model=EmotionResponse)
@limiter.limit(f"{settings.rate_limit_requests}/minute")
async def detect_emotion(
    request: Request,
    emotion_request: EmotionRequest,
    detector: EmotionDetectorService = Depends(get_detector)
):
    """
    Detect emotion from a base64 encoded image
    
    - **image**: Base64 encoded image (JPEG/PNG)
    - **face_bounds**: Optional face bounding box {x, y, width, height}
    - **return_all_scores**: Return confidence scores for all emotions
    """
    try:
        # Decode image
        image_bytes = base64.b64decode(emotion_request.image)
        
        # Check image size
        if len(image_bytes) > settings.max_image_size:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Image size exceeds {settings.max_image_size} bytes"
            )
        
        # Detect emotion
        result = await detector.detect_emotion(
            image_bytes,
            face_bounds=emotion_request.face_bounds
        )
        
        return EmotionResponse(
            success=result.face_detected,
            emotion=result.emotion,
            confidence=result.confidence,
            all_scores=result.all_scores if emotion_request.return_all_scores else None,
            processing_time=result.processing_time,
            model_used=result.model_used,
            face_detected=result.face_detected,
            landmarks=result.landmarks,
            timestamp=time.time()
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Detection error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/api/detect/upload")
@limiter.limit(f"{settings.rate_limit_requests}/minute")
async def detect_emotion_upload(
    request: Request,
    file: UploadFile = File(...),
    detector: EmotionDetectorService = Depends(get_detector)
):
    """
    Detect emotion from an uploaded image file
    """
    try:
        # Validate file type
        if file.content_type not in ["image/jpeg", "image/png", "image/webp"]:
            raise HTTPException(
                status_code=400,
                detail="Invalid file type. Only JPEG, PNG, and WebP are supported"
            )
        
        # Read file
        contents = await file.read()
        
        # Check file size
        if len(contents) > settings.max_image_size:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Image size exceeds {settings.max_image_size} bytes"
            )
        
        # Detect emotion
        result = await detector.detect_emotion(contents)
        
        return EmotionResponse(
            success=result.face_detected,
            emotion=result.emotion,
            confidence=result.confidence,
            all_scores=result.all_scores,
            processing_time=result.processing_time,
            model_used=result.model_used,
            face_detected=result.face_detected,
            landmarks=result.landmarks,
            timestamp=time.time()
        )
        
    except Exception as e:
        logger.error(f"Upload detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/detect/batch")
@limiter.limit("10/minute")
async def detect_batch(
    request: Request,
    images: List[str],
    detector: EmotionDetectorService = Depends(get_detector)
):
    """
    Batch emotion detection (max 10 images)
    """
    if len(images) > settings.max_batch_size:
        raise HTTPException(
            status_code=400,
            detail=f"Batch size exceeds maximum of {settings.max_batch_size}"
        )
    
    try:
        # Decode all images
        image_bytes_list = []
        for img_str in images:
            if ',' in img_str:  # Handle data URL
                img_str = img_str.split(',')[1]
            image_bytes_list.append(base64.b64decode(img_str))
        
        # Process batch
        results = await detector.batch_detect(image_bytes_list)
        
        # Format responses
        responses = []
        for result in results:
            responses.append({
                "emotion": result.emotion,
                "confidence": result.confidence,
                "all_scores": result.all_scores,
                "face_detected": result.face_detected
            })
        
        return {"success": True, "results": responses}
        
    except Exception as e:
        logger.error(f"Batch detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# WebSocket for real-time detection
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time emotion detection
    Send base64 encoded frames and receive emotion results
    """
    await websocket.accept()
    app.state.connections.add(websocket)
    detector = getattr(app.state, 'detector', None)
    
    if not detector:
        await websocket.send_json({
            "type": "error",
            "data": {"error": "Emotion detector not available"}
        })
        await websocket.close()
        return
    
    try:
        while True:
            # Receive frame data
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message["type"] == "frame":
                try:
                    # Decode and process frame
                    image_b64 = message["data"]["image"]
                    if ',' in image_b64:
                        image_b64 = image_b64.split(',')[1]
                    
                    image_bytes = base64.b64decode(image_b64)
                    face_bounds = message.get("data", {}).get("face_bounds")
                    
                    # Detect emotion
                    result = await detector.detect_emotion(image_bytes, face_bounds)
                    
                    # Send result back
                    await websocket.send_json({
                        "type": "result",
                        "data": result.to_dict()
                    })
                    
                except Exception as e:
                    await websocket.send_json({
                        "type": "error",
                        "data": {"error": str(e)}
                    })
            
            elif message["type"] == "ping":
                await websocket.send_json({"type": "pong"})
                
    except WebSocketDisconnect:
        app.state.connections.discard(websocket)
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        app.state.connections.discard(websocket)


# Serve HTML client
@app.get("/", response_class=HTMLResponse)
async def get_client():
    """Serve the web client"""
    try:
        static_file = Path("static/index.html")
        if static_file.exists():
            with open(static_file, "r", encoding="utf-8") as f:
                return HTMLResponse(content=f.read())
        else:
            return HTMLResponse(content="""
            <html>
                <head><title>Facial Emotion Detector</title></head>
                <body>
                    <h1>Facial Emotion Detector API</h1>
                    <p>The API is running! However, the static HTML file is not found.</p>
                    <p>Available endpoints:</p>
                    <ul>
                        <li><a href="/health">/health</a> - Health check</li>
                        <li><a href="/api/status">/api/status</a> - API status</li>
                        <li><a href="/api/docs">/api/docs</a> - API documentation (if debug enabled)</li>
                    </ul>
                </body>
            </html>
            """)
    except Exception as e:
        logger.error(f"Error serving client: {e}")
        return HTMLResponse(content=f"<h1>Error loading client: {e}</h1>")


if __name__ == "__main__":
    # Run with production settings
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        workers=settings.workers,
        log_level=settings.log_level,
        access_log=True,
        use_colors=True,
        # Optimize for low memory
        limit_concurrency=100,
        limit_max_requests=1000,
        timeout_keep_alive=5
    )