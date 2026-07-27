# 🚀 DỰ ÁN RESNET50 VISUALIZER & CLASSIFIER

Chào mừng bạn đến với dự án trực quan hóa chuyên sâu mô hình **ResNet50**. Dự án không chỉ giúp phân tích, bóc tách các lớp ẩn (Feature Maps, Grad-CAM) của mạng Neural mà còn tích hợp khả năng nhận diện đa mô hình.

Dự án bao gồm hai thành phần chính: 
- **Backend (Python - FastAPI)** xử lý AI và suy luận (Inference).
- **Frontend (React - Vite)** giao diện tương tác người dùng (Dashboard).

---

## ✨ Tính năng nổi bật
1. **Hỗ trợ Đa mô hình (Multi-model)**: 
   - 🌍 *Mô hình ImageNet*: Nhận diện "vạn vật" với 1000 nhãn phân loại khác nhau.
   - 🐶🐱 *Mô hình Custom (Chó/Mèo)*: Mô hình ResNet50 được Fine-tuning riêng để phân biệt giữa chó và mèo với độ chính xác cao.
2. **Chuyển đổi Mô hình Thời gian thực**: Cả 2 mô hình được tải sẵn vào RAM/VRAM giúp việc chuyển đổi và nhận diện trên giao diện diễn ra tức thì.
3. **Phân tích Sâu (Deep Analysis)**:
   - 🔬 *Feature Maps*: Trực quan hóa đầu ra của từng Layer/Stage.
   - 📈 *Gradient Analysis*: Phân tích sự lan truyền ngược của tín hiệu gradient.
   - 🔥 *Grad-CAM Heatmap*: Xác định vùng không gian trên ảnh có tác động lớn nhất tới quyết định của mô hình.

---

## 🛠️ Yêu cầu hệ thống (Prerequisites)
Để khởi chạy dự án, máy tính của bạn cần cài đặt:
- **Python 3.10+**: Môi trường chạy FastAPI và PyTorch.
- **Node.js (LTS)**: Môi trường chạy ReactJS.

---

## 📂 Cấu trúc dự án
```text
Resnet/
├── backend/            # Chứa mã nguồn API, xử lý model PyTorch
│   ├── weights/        # Chứa file weights (.pth) của mô hình (ví dụ: resnet50_catsdogs.pth)
│   ├── main.py         # Điểm vào (Entry point) của FastAPI
│   └── model_handler.py# Xử lý load model, hooks và trích xuất đặc trưng
└── frontend/           # Chứa mã nguồn React giao diện người dùng
    ├── src/            
    ├── package.json    
    └── index.html      
```

---

## 1. Khởi chạy Backend (Máy chủ AI)

Backend chịu trách nhiệm chạy các mạng nơ-ron, trích xuất Feature Maps và tính toán Grad-CAM.

1. **Mở Terminal** và di chuyển vào thư mục `backend`:
   ```bash
   cd backend
   ```
2. **Cài đặt các thư viện cần thiết**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Khởi động Server**:
   ```bash
   python main.py
   ```
   *Mặc định Server sẽ chạy tại: `http://localhost:8000`*

> **Lưu ý:** Ở lần chạy đầu tiên, Backend sẽ tự động tải trọng số (Weights) của ResNet50 từ máy chủ PyTorch (khoảng 98MB) cho mô hình ImageNet. Mô hình Chó/Mèo sẽ được load trực tiếp từ file `backend/weights/resnet50_catsdogs.pth`.

---

## 2. Khởi chạy Frontend (Giao diện Studio)

1. **Mở một Terminal mới** và di chuyển vào thư mục `frontend`:
   ```bash
   cd frontend
   ```
2. **Cài đặt Dependencies** (Chỉ cần chạy ở lần đầu):
   ```bash
   npm install
   ```
3. **Khởi chạy Dev Server**:
   ```bash
   npm run dev
   ```
   *Giao diện sẽ hiển thị tại: `http://localhost:5173`*

---

## 🎨 Cách sử dụng Dashboard

1. Truy cập vào trình duyệt theo địa chỉ `http://localhost:5173`.
2. Sử dụng thanh công cụ bên trái (**Sidebar**) để tải lên hình ảnh bạn muốn kiểm tra.
3. Chọn mô hình bạn muốn dùng ở mục **🧠 Chọn Mô hình** (ImageNet hoặc Chó Mèo).
4. Hệ thống sẽ tự động chạy ảnh qua 50 layers của ResNet50 và hiển thị:
   - Luồng chạy (Pipeline Flow) qua từng Node.
   - Bảng phân tích chi tiết (Pixel-level, Pooling Analysis, Feature Maps).
   - Biểu đồ xác suất dự đoán (Confidence).

**Chúc bạn có những trải nghiệm khám phá kiến trúc ResNet50 thú vị!**
