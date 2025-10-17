from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import numpy as np
import cv2
from typing import List
import logging
import os

from ..models.ensemble_model import EnsembleMedicalModel
from ..preprocessor.image_loader import MedicalImageLoader
from ..preprocessor.enhancement import MedicalImageEnhancer
from ..features.feature_fusion import FeatureFusion

logger = logging.getLogger(__name__)

def create_app(config: dict) -> FastAPI:
    """Create FastAPI application"""
    app = FastAPI(
        title="MedVision AI Agent API",
        description="Medical Image Analysis AI System",
        version="1.0.0"
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Initialize components
    model_ensemble = EnsembleMedicalModel(config)
    image_loader = MedicalImageLoader()
    image_enhancer = MedicalImageEnhancer()
    feature_extractor = FeatureFusion()
    
    @app.get("/")
    async def root():
        return {"message": "MedVision AI Agent API", "version": "1.0.0"}
    
    @app.get("/health")
    async def health_check():
        return {"status": "healthy", "timestamp": "2024-01-15T10:30:00Z"}
    
    @app.post("/analyze/single")
    async def analyze_single_image(
        image: UploadFile = File(...),
        model_type: str = "ensemble",
        return_attention: bool = False
    ):
        try:
            # Read and preprocess image
            contents = await image.read()
            nparr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                raise HTTPException(status_code=400, detail="Could not decode image")
            
            # Preprocess image
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            enhanced_img = image_enhancer.enhance_contrast(img_rgb)
            
            # TODO: Add model prediction logic
            prediction = {
                "class": "normal",
                "confidence": 0.85,
                "probabilities": {"normal": 0.85, "abnormal": 0.15}
            }
            
            return {
                "prediction": prediction,
                "analysis": {
                    "processing_time": 0.5,
                    "model_used": model_type,
                    "image_size": list(img.shape)
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/analyze/batch")
    async def analyze_batch_images(images: List[UploadFile] = File(...)):
        try:
            results = []
            
            for image in images:
                contents = await image.read()
                nparr = np.frombuffer(contents, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if img is not None:
                    # TODO: Add batch processing logic
                    results.append({
                        "filename": image.filename,
                        "prediction": "normal",
                        "confidence": 0.85,
                        "processing_time": 0.5
                    })
            
            return {
                "results": results,
                "summary": {
                    "total_images": len(results),
                    "successful_analysis": len(results)
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/models")
    async def get_models_info():
        return {
            "available_models": [
                {
                    "name": "lightweight_cnn",
                    "type": "cnn",
                    "status": "ready",
                    "input_shape": [224, 224, 3]
                }
            ]
        }
    
    return app

def start_rest_api(host: str = "0.0.0.0", port: int = 8000, config: dict = None):
    """Start REST API server"""
    app = create_app(config or {})
    
    logger.info(f"🚀 Starting MedVision AI API server on {host}:{port}")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )
