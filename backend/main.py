"""
main.py
=======
FastAPI server phục vụ API cho ứng dụng ResNet50 Visualization.

Endpoints:
  POST /predict   – Nhận ảnh, trả kết quả dự đoán + feature maps + gradients.
  GET  /health    – Kiểm tra trạng thái server.
  GET  /model-info – Trả thông tin tổng quan về kiến trúc ResNet50.
"""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from model_handler import ResNet50Handler

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Khởi tạo model handler (singleton)
# ---------------------------------------------------------------------------
handlers = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model khi server khởi động, giải phóng khi tắt."""
    global handlers
    logger.info("🚀 Đang tải mô hình ResNet50 (ImageNet)...")
    handlers["imagenet"] = ResNet50Handler(model_type="imagenet")
    logger.info("🚀 Đang tải mô hình ResNet50 (Cats/Dogs)...")
    handlers["catsdogs"] = ResNet50Handler(model_type="catsdogs")
    logger.info("✅ Models đã sẵn sàng")
    yield
    logger.info("🛑 Server đang tắt...")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="ResNet50 Visualization API",
    description="API trích xuất feature maps, gradients và dự đoán từ ResNet50.",
    version="1.0.0",
    lifespan=lifespan,
)

# Cho phép CORS từ frontend (Electron / Vite dev server)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
async def health_check():
    """Kiểm tra server còn sống."""
    return {"status": "ok", "models_loaded": len(handlers) > 0}


@app.get("/model-info")
async def model_info():
    """Trả về thông tin kiến trúc ResNet50 để frontend vẽ pipeline."""
    return {
        "model_name": "ResNet50",
        "input_size": [3, 224, 224],
        "num_classes": 1000,
        "pipeline": [
            {
                "id": "input",
                "label": "Input Image",
                "type": "input",
                "description": "Ảnh gốc được resize về 224×224 và normalize theo chuẩn ImageNet.",
            },
            {
                "id": "conv1",
                "label": "Conv 7×7",
                "type": "conv",
                "description": "Tích chập đầu tiên: kernel 7×7, stride 2, 64 filters. Giảm kích thước xuống 112×112.",
            },
            {
                "id": "bn1",
                "label": "BatchNorm",
                "type": "norm",
                "description": "Chuẩn hóa batch giúp training ổn định hơn.",
            },
            {
                "id": "relu",
                "label": "ReLU",
                "type": "activation",
                "description": "Hàm kích hoạt phi tuyến: f(x) = max(0, x).",
            },
            {
                "id": "maxpool",
                "label": "Max Pooling 3×3",
                "type": "pool",
                "description": "Giảm kích thước spatial xuống 56×56, giữ lại đặc trưng mạnh nhất.",
            },
            {
                "id": "stage1",
                "label": "Stage 1",
                "type": "stage",
                "blocks": "1 Conv Block + 2 Identity Blocks",
                "output_filters": 256,
                "description": "Trích xuất edges và textures – đặc trưng cơ bản.",
            },
            {
                "id": "stage2",
                "label": "Stage 2",
                "type": "stage",
                "blocks": "1 Conv Block + 3 Identity Blocks",
                "output_filters": 512,
                "description": "Nhận diện shapes – pattern đơn giản.",
            },
            {
                "id": "stage3",
                "label": "Stage 3",
                "type": "stage",
                "blocks": "1 Conv Block + 5 Identity Blocks",
                "output_filters": 1024,
                "description": "Phát hiện đối tượng phức tạp hơn.",
            },
            {
                "id": "stage4",
                "label": "Stage 4",
                "type": "stage",
                "blocks": "1 Conv Block + 2 Identity Blocks",
                "output_filters": 2048,
                "description": "Semantic features – đặc trưng mang ý nghĩa trừu tượng cao.",
            },
            {
                "id": "avgpool",
                "label": "Global Avg Pooling",
                "type": "pool",
                "description": "Nén feature map thành vector 1D (2048,).",
            },
            {
                "id": "fc",
                "label": "Fully Connected (Softmax)",
                "type": "fc",
                "description": "Layer cuối cùng trả về phân phối xác suất trên 1000 lớp ImageNet.",
            },
        ],
    }


@app.post("/predict")
def predict(file: UploadFile = File(...), model_type: str = Form("imagenet")):
    """
    Nhận file ảnh upload, chạy ResNet50 inference và trả kết quả chi tiết.

    Returns:
        JSON chứa prediction, confidence, top5, feature maps, gradients, Grad-CAM.
    """
    handler = handlers.get(model_type)
    if handler is None:
        raise HTTPException(status_code=503, detail=f"Model {model_type} chưa được tải.")

    # Kiểm tra loại file
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"File phải là ảnh (nhận được: {file.content_type}).",
        )

    try:
        file_bytes = file.file.read()
        if len(file_bytes) == 0:
            raise HTTPException(status_code=400, detail="File ảnh rỗng.")

        logger.info(f"📷 Đang xử lý ảnh: {file.filename} ({len(file_bytes)} bytes)")
        start = time.time()

        result = handler.predict(file_bytes)

        elapsed = time.time() - start
        logger.info(
            f"✅ Kết quả: {result['prediction']} "
            f"({result['confidence']:.2%}) – {elapsed:.2f}s"
        )

        return JSONResponse(content=result)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("❌ Lỗi khi xử lý ảnh")
        raise HTTPException(status_code=500, detail=f"Lỗi server: {str(e)}")


# ---------------------------------------------------------------------------
# Run trực tiếp
# ---------------------------------------------------------------------------
@app.get("/api/residual/{stage}/{block_index}")
def get_residual(stage: str, block_index: int, model_type: str = "imagenet"):
    handler = handlers.get(model_type)
    if not handler:
        raise HTTPException(status_code=404, detail="Model not found")
    data = handler.get_residual_data(stage, block_index)
    if not data:
        raise HTTPException(status_code=404, detail="Block not found or image not cached")
    return data

@app.get("/api/receptive_field")
def get_receptive_field(layer: str, pixel_x: int, pixel_y: int, model_type: str = "imagenet"):
    handler = handlers.get(model_type)
    if not handler:
        raise HTTPException(status_code=404, detail="Model not found")
    return handler.get_receptive_field(layer, pixel_x, pixel_y)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
