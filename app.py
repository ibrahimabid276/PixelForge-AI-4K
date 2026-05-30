from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
import uuid
import os

# Patch basicsr compatibility issue
import patch_basicsr

from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

app = FastAPI(title="PixelForge AI 4K Upscaler")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
WEIGHTS_DIR = "weights"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(WEIGHTS_DIR, exist_ok=True)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Running on: {device}")

# Initialize the RRDBNet model architecture
model_arch = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32)

# Initialize RealESRGANer with the model
model = RealESRGANer(
    scale=4,
    model_path="weights/RealESRGAN_x4plus.pth",
    model=model_arch,
    device=device,
    half=False  # Don't use half precision on CPU
)

@app.get("/")
def root():
    return {
        "app": "PixelForge AI",
        "status": "running",
        "upscaler": "Real-ESRGAN x4"
    }

@app.post("/upscale")
async def upscale_image(file: UploadFile = File(...)):
    try:

        image_id = str(uuid.uuid4())

        input_path = os.path.join(
            UPLOAD_DIR,
            f"{image_id}.png"
        )

        output_path = os.path.join(
            OUTPUT_DIR,
            f"{image_id}.png"
        )

        with open(input_path, "wb") as buffer:
            buffer.write(await file.read())

        image = Image.open(
            input_path
        ).convert("RGB")

        # Convert PIL image to numpy array
        import numpy as np
        img_np = np.array(image)

        # Enhance image using RealESRGANer
        _, sr_img, _ = model.enhance(img_np, outscale=4)

        # Convert back to PIL Image
        from PIL import Image
        sr_image = Image.fromarray(sr_img)

        sr_image.save(
            output_path,
            quality=100
        )

        return {
            "success": True,
            "image_id": image_id,
            "download_url": f"/download/{image_id}"
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e)
            }
        )

@app.get("/download/{image_id}")
def download_image(image_id: str):

    file_path = os.path.join(
        OUTPUT_DIR,
        f"{image_id}.png"
    )

    if not os.path.exists(file_path):
        return JSONResponse(
            status_code=404,
            content={
                "success": False,
                "message": "Image not found"
            }
        )

    return FileResponse(
        file_path,
        media_type="image/png",
        filename="PixelForge-4K.png"
    )

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )