"""
FastAPI server for Qwen VLM inference.

Run on remote GPU server (e.g., H100):
    python -m models.qwen_server --host 0.0.0.0 --port 8000 --batch-size 12

Or with uvicorn directly:
    uvicorn models.qwen_server:app --host 0.0.0.0 --port 8000
"""

import base64
import io
import argparse
from typing import List, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
import uvicorn

# Lazy imports for GPU dependencies
_model = None
_processor = None


class GenerateRequest(BaseModel):
    """Single image generation request."""
    image_base64: str
    prompt: str
    max_tokens: int = 512


class BatchGenerateRequest(BaseModel):
    """Batch generation request."""
    images_base64: List[str]
    prompts: List[str]
    max_tokens: int = 512


class GenerateResponse(BaseModel):
    """Generation response."""
    response: str
    success: bool = True
    error: Optional[str] = None


class BatchGenerateResponse(BaseModel):
    """Batch generation response."""
    responses: List[str]
    success: bool = True
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    model_name: Optional[str] = None
    device: Optional[str] = None


app = FastAPI(
    title="Qwen VLM Server",
    description="Remote inference server for Qwen Vision-Language Model",
    version="1.0.0"
)


def get_model():
    """Get or initialize the model."""
    global _model
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Start server with model loading.")
    return _model


def decode_image(image_base64: str) -> Image.Image:
    """Decode base64 image to PIL Image."""
    try:
        image_bytes = base64.b64decode(image_base64)
        return Image.open(io.BytesIO(image_bytes)).convert('RGB')
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {e}")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check server health and model status."""
    global _model
    if _model is None:
        return HealthResponse(status="running", model_loaded=False)
    return HealthResponse(
        status="ready",
        model_loaded=True,
        model_name=_model.model_name,
        device=_model.device
    )


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """Generate response for a single image."""
    try:
        model = get_model()
        image = decode_image(request.image_base64)

        response = model.generate(
            image=image,
            prompt=request.prompt,
            max_tokens=request.max_tokens
        )

        return GenerateResponse(response=response)
    except HTTPException:
        raise
    except Exception as e:
        return GenerateResponse(response="", success=False, error=str(e))


@app.post("/batch_generate", response_model=BatchGenerateResponse)
async def batch_generate(request: BatchGenerateRequest):
    """Generate responses for multiple images (batched)."""
    try:
        model = get_model()

        if len(request.images_base64) != len(request.prompts):
            raise HTTPException(
                status_code=400,
                detail=f"Number of images ({len(request.images_base64)}) must match prompts ({len(request.prompts)})"
            )

        # Decode all images
        images = [decode_image(img_b64) for img_b64 in request.images_base64]

        # Use batched generation
        if hasattr(model, 'batch_generate'):
            responses = model.batch_generate(
                images=images,
                prompts=request.prompts,
                max_tokens=request.max_tokens
            )
        else:
            # Fallback to sequential
            responses = [
                model.generate(img, prompt, request.max_tokens)
                for img, prompt in zip(images, request.prompts)
            ]

        return BatchGenerateResponse(responses=responses)
    except HTTPException:
        raise
    except Exception as e:
        return BatchGenerateResponse(responses=[], success=False, error=str(e))


def load_model(model_size: str = "8B", device: str = "cuda", batch_size: int = 12):
    """Load the Qwen model."""
    global _model

    print(f"Loading Qwen VLM ({model_size}) on {device}...")

    from .qwen_vlm import QwenVLM

    model_names = {
        "8B": "Qwen/Qwen3-VL-8B-Instruct",
        "2B": "Qwen/Qwen3-VL-2B-Instruct"
    }

    model_name = model_names.get(model_size, model_size)

    _model = QwenVLM(
        model_name=model_name,
        device=device,
        enable_batching=True
    )
    _model._max_batch_size = batch_size

    print(f"Model loaded! Max batch size: {batch_size}")
    return _model


def main():
    """Run the server."""
    parser = argparse.ArgumentParser(description="Qwen VLM Inference Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    parser.add_argument("--model-size", type=str, default="8B", choices=["8B", "2B"], help="Model size")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--batch-size", type=int, default=12, help="Max batch size")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")

    args = parser.parse_args()

    # Load model before starting server
    load_model(args.model_size, args.device, args.batch_size)

    # Run server
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=args.workers
    )


if __name__ == "__main__":
    main()
