"""
model_handler.py
================
Module xử lý mô hình ResNet50 pre-trained.
- Đăng ký forward hooks tại từng stage để trích xuất feature maps.
- Tính toán Grad-CAM heatmap.
- Trả về kết quả dự đoán kèm dữ liệu visualization dạng Base64.
"""

import io
import base64
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

# ---------------------------------------------------------------------------
# Hằng số
# ---------------------------------------------------------------------------
# Số lượng feature maps tối đa xuất ra cho mỗi layer (giảm tải cho frontend)
MAX_FEATURE_MAPS = 20

# ImageNet class labels – file sẽ được tải lần đầu khi cần
_IMAGENET_LABELS: Optional[List[str]] = None

# ---------------------------------------------------------------------------
# Tiền xử lý ảnh chuẩn ImageNet
# ---------------------------------------------------------------------------
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def load_imagenet_labels() -> List[str]:
    """Trả về danh sách 1000 labels ImageNet."""
    global _IMAGENET_LABELS
    if _IMAGENET_LABELS is not None:
        return _IMAGENET_LABELS

    # Sử dụng torchvision meta nếu có, fallback sang danh sách mặc định
    try:
        from torchvision.models import ResNet50_Weights
        weights = ResNet50_Weights.IMAGENET1K_V2
        _IMAGENET_LABELS = weights.meta["categories"]
    except Exception:
        # Fallback: tạo danh sách class_0 .. class_999
        _IMAGENET_LABELS = [f"class_{i}" for i in range(1000)]
    return _IMAGENET_LABELS


# ---------------------------------------------------------------------------
# Class chính
# ---------------------------------------------------------------------------
class ResNet50Handler:
    """Quản lý mô hình ResNet50 và trích xuất dữ liệu visualization."""

    # Tên các nhóm layer chính cần hook
    HOOK_TARGETS = {
        "input_preprocess": None,        # Ảnh sau khi resize/normalize (không cần hook)
        "conv1":            "conv1",      # Conv 7x7
        "bn1":              "bn1",
        "relu":             "relu",
        "maxpool":          "maxpool",    # Max Pooling
        "stage1":           "layer1",     # Stage 1
        "stage2":           "layer2",     # Stage 2
        "stage3":           "layer3",     # Stage 3
        "stage4":           "layer4",     # Stage 4
        "avgpool":          "avgpool",    # Global Average Pooling
    }

    def __init__(self, model_type: str = "imagenet"):
        import os
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.resnet50()
        self.model_type = model_type
        
        if model_type == "catsdogs":
            local_weights = os.path.join(os.path.dirname(__file__), "weights", "resnet50_catsdogs.pth")
            if os.path.exists(local_weights):
                print(f"Loading local model: {local_weights}")
                # Customize fc layer for 2 classes
                self.model.fc = torch.nn.Linear(self.model.fc.in_features, 2)
                self.model.load_state_dict(torch.load(local_weights, map_location=self.device))
                self.labels = ["Cat", "Dog"]
            else:
                print(f"Warning: {local_weights} not found. Fallback to ImageNet.")
                self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
                self.labels = load_imagenet_labels()
        else:
            print("Loading pre-trained ImageNet model")
            self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            self.labels = load_imagenet_labels()
            
        self.model.to(self.device)
        self.model.eval()

        # Tắt inplace ReLU để tránh lỗi autograd với backward hooks
        self._disable_inplace_relu(self.model)

        # Lưu trữ output của từng layer khi forward pass
        self._features: Dict[str, torch.Tensor] = {}
        # Lưu gradient cho Grad-CAM
        self._gradients: Dict[str, torch.Tensor] = {}

        # Đăng ký hooks
        self._register_hooks()

    # ----- Hook helpers ----------------------------------------------------

    @staticmethod
    def _disable_inplace_relu(model):
        """Tắt inplace trên tất cả ReLU trong model để backward hooks hoạt động đúng."""
        for module in model.modules():
            if isinstance(module, torch.nn.ReLU):
                module.inplace = False

    def _register_hooks(self):
        """Gắn forward & backward hooks vào các layer quan trọng."""
        for label, attr_name in self.HOOK_TARGETS.items():
            if attr_name is None:
                continue
            layer = getattr(self.model, attr_name, None)
            if layer is None:
                continue

            # Forward hook: lưu output
            layer.register_forward_hook(self._make_forward_hook(label))
            # Backward hook: lưu gradient (dùng cho Grad-CAM)
            layer.register_full_backward_hook(self._make_backward_hook(label))

    def _make_forward_hook(self, name: str):
        def hook(_module, _input, output):
            self._features[name] = output.clone().detach()
        return hook

    def _make_backward_hook(self, name: str):
        def hook(_module, _grad_input, grad_output):
            self._gradients[name] = grad_output[0].clone().detach()
        return hook

    # ----- Xử lý ảnh ------------------------------------------------------

    @staticmethod
    def _pil_from_bytes(file_bytes: bytes) -> Image.Image:
        """Chuyển bytes thành PIL Image (RGB)."""
        return Image.open(io.BytesIO(file_bytes)).convert("RGB")

    @staticmethod
    def _tensor_to_base64_images(
        tensor: torch.Tensor,
        max_maps: int = MAX_FEATURE_MAPS,
    ) -> List[str]:
        """
        Chuyển tensor feature-map (1, C, H, W) thành danh sách ảnh PNG Base64.
        Chỉ lấy tối đa `max_maps` channels đầu tiên.
        """
        if tensor.dim() == 4:
            tensor = tensor[0]  # bỏ batch dim -> (C, H, W)

        c = tensor.shape[0]
        indices = list(range(min(c, max_maps)))
        images: List[str] = []

        for idx in indices:
            feat = tensor[idx].cpu().numpy()
            # Normalize về [0, 255]
            feat -= feat.min()
            max_val = feat.max()
            if max_val > 0:
                feat = feat / max_val
            feat = (feat * 255).astype(np.uint8)
            # Áp dụng colormap cho dễ nhìn
            colored = cv2.applyColorMap(feat, cv2.COLORMAP_VIRIDIS)
            _, buf = cv2.imencode(".png", colored)
            b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
            images.append(b64)

        return images

    @staticmethod
    def _image_to_base64(img_array: np.ndarray) -> str:
        """Chuyển numpy array (H,W,3 BGR) thành Base64 PNG."""
        _, buf = cv2.imencode(".png", img_array)
        return base64.b64encode(buf.tobytes()).decode("utf-8")

    # ----- Grad-CAM --------------------------------------------------------

    def _compute_gradcam(self, target_layer: str, class_idx: int) -> np.ndarray:
        """
        Tính Grad-CAM heatmap cho `target_layer` ứng với class `class_idx`.
        Trả về heatmap (H, W) đã normalize [0, 255] uint8.
        """
        grads = self._gradients.get(target_layer)
        feats = self._features.get(target_layer)

        if grads is None or feats is None:
            return np.zeros((7, 7), dtype=np.uint8)

        # Global average pooling trên gradient
        weights = grads.mean(dim=[2, 3], keepdim=True)  # (1, C, 1, 1)
        cam = (weights * feats).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        # Normalize
        cam -= cam.min()
        max_val = cam.max()
        if max_val > 0:
            cam = cam / max_val
        cam = (cam * 255).astype(np.uint8)
        return cam

    # ----- API chính -------------------------------------------------------

    def predict(self, file_bytes: bytes) -> dict:
        """
        Chạy inference trên ảnh, trả về dict chứa:
        - prediction, confidence
        - layers: danh sách thông tin + feature maps (base64)
        - gradcam: heatmap base64
        - gradient_magnitudes: độ lớn gradient trung bình mỗi layer
        """
        # ---- Bước 1: Tiền xử lý ảnh ----
        self.last_image_bytes = file_bytes
        pil_img = self._pil_from_bytes(file_bytes)
        input_tensor = preprocess(pil_img).unsqueeze(0).to(self.device)
        input_tensor.requires_grad_(True)

        # Lưu ảnh gốc đã resize 224x224
        resized = pil_img.resize((224, 224))
        original_np = cv2.cvtColor(np.array(resized), cv2.COLOR_RGB2BGR)

        # Reset caches
        self._features.clear()
        self._gradients.clear()

        # ---- Bước 2: Forward pass ----
        output = self.model(input_tensor)
        probs = F.softmax(output, dim=1)
        confidence, pred_idx = probs.max(dim=1)
        pred_idx = pred_idx.item()
        confidence = confidence.item()

        # ---- Bước 3: Backward pass (để lấy gradient cho Grad-CAM) ----
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, pred_idx] = 1.0
        output.backward(gradient=one_hot, retain_graph=True)

        # ---- Bước 4: Lấy label ----
        labels = self.labels
        pred_label = labels[pred_idx] if pred_idx < len(labels) else f"class_{pred_idx}"

        # ---- Bước 5: Top-5 predictions ----
        k = min(5, len(labels))
        topk_probs, topk_indices = probs[0].topk(k)
        top5 = []
        for p, idx in zip(topk_probs.tolist(), topk_indices.tolist()):
            lbl = labels[idx] if idx < len(labels) else f"class_{idx}"
            top5.append({"label": lbl, "confidence": round(p, 4)})

        # ---- Bước 6: Xây dựng dữ liệu layers ----
        layer_meta = {
            "conv1":   {"type": "Conv2D",  "kernel": "7x7", "stride": 2, "filters": 64,
                        "role": "Trích xuất đặc trưng thô ban đầu từ ảnh gốc"},
            "bn1":     {"type": "BatchNorm", "kernel": "-", "stride": "-", "filters": 64,
                        "role": "Chuẩn hóa phân phối activation"},
            "relu":    {"type": "ReLU",    "kernel": "-", "stride": "-", "filters": 64,
                        "role": "Kích hoạt phi tuyến, loại bỏ giá trị âm"},
            "maxpool": {"type": "MaxPool2D", "kernel": "3x3", "stride": 2, "filters": 64,
                        "role": "Giảm chiều không gian, giữ đặc trưng mạnh nhất"},
            "stage1":  {"type": "ResNet Stage", "kernel": "Mixed", "stride": 1, "filters": 256,
                        "role": "Edges & textures – đặc trưng cơ bản"},
            "stage2":  {"type": "ResNet Stage", "kernel": "Mixed", "stride": 2, "filters": 512,
                        "role": "Shapes – pattern đơn giản"},
            "stage3":  {"type": "ResNet Stage", "kernel": "Mixed", "stride": 2, "filters": 1024,
                        "role": "Đối tượng phức tạp hơn"},
            "stage4":  {"type": "ResNet Stage", "kernel": "Mixed", "stride": 2, "filters": 2048,
                        "role": "Semantic features – ý nghĩa trừu tượng cao"},
            "avgpool": {"type": "AdaptiveAvgPool2D", "kernel": "Global", "stride": "-", "filters": 2048,
                        "role": "Nén toàn bộ spatial dimensions thành vector"},
        }

        layers_output: List[dict] = []
        gradient_magnitudes: Dict[str, float] = {}

        for label in ["conv1", "bn1", "relu", "maxpool",
                       "stage1", "stage2", "stage3", "stage4", "avgpool"]:
            feat = self._features.get(label)
            if feat is None:
                continue
            shape = list(feat.shape)
            meta = layer_meta.get(label, {})

            # Feature maps base64
            feature_maps_b64 = []
            if feat.dim() == 4 and feat.shape[2] > 1 and feat.shape[3] > 1:
                feature_maps_b64 = self._tensor_to_base64_images(feat)

            # Gradient magnitude trung bình
            grad = self._gradients.get(label)
            grad_mag = 0.0
            if grad is not None:
                grad_mag = grad.abs().mean().item()
            gradient_magnitudes[label] = round(grad_mag, 6)

            feature_values = []
            if label == "avgpool" and feat is not None:
                # feat shape: (1, 2048, 1, 1) -> flatten to list
                feature_values = feat.view(-1).cpu().detach().numpy().tolist()

            layers_output.append({
                "name":          label,
                "type":          meta.get("type", "Unknown"),
                "kernel":        meta.get("kernel", "-"),
                "stride":        meta.get("stride", "-"),
                "filters":       meta.get("filters", 0),
                "output_shape":  shape,
                "role":          meta.get("role", ""),
                "feature_maps":  feature_maps_b64,
                "pooled_values":  feature_values, # Added this
                "gradient_mag":  grad_mag,
            })

        # ---- Bước 7: Grad-CAM heatmap trên stage4 ----
        cam = self._compute_gradcam("stage4", pred_idx)
        cam_resized = cv2.resize(cam, (224, 224))
        heatmap_color = cv2.applyColorMap(cam_resized, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(original_np, 0.5, heatmap_color, 0.5, 0)
        gradcam_b64 = self._image_to_base64(overlay)
        heatmap_only_b64 = self._image_to_base64(heatmap_color)

        # ---- Bước 8: Ảnh gốc base64 ----
        original_b64 = self._image_to_base64(original_np)

        return {
            "prediction":           pred_label,
            "confidence":           round(confidence, 4),
            "top5":                 top5,
            "original_image":       original_b64,
            "gradcam_overlay":      gradcam_b64,
            "gradcam_heatmap":      heatmap_only_b64,
            "layers":               layers_output,
            "gradient_magnitudes":  gradient_magnitudes,
        }

    # ----- Tính toán Residual f(x) vs f(x)+x -------------------
    def get_residual_data(self, stage: str, block_index: int) -> dict:
        """
        Lấy thông tin của f(x) (trước skip) và f(x)+x (sau skip).
        """
        if not hasattr(self, 'last_image_bytes'):
            return {}
        file_bytes = self.last_image_bytes
        stage_module = getattr(self.model, stage.replace("stage", "layer"), None)
        if stage_module is None or block_index >= len(stage_module):
            return {}

        block = stage_module[block_index]
        
        fx_tensors = {}
        out_tensors = {}

        def fx_hook(m, inp, out):
            fx_tensors["fx"] = out.clone().detach()
        def out_hook(m, inp, out):
            out_tensors["out"] = out.clone().detach()

        h1 = block.bn3.register_forward_hook(fx_hook)
        h2 = block.register_forward_hook(out_hook)

        pil_img = self._pil_from_bytes(file_bytes)
        input_tensor = preprocess(pil_img).unsqueeze(0).to(self.device)
        self.model(input_tensor)

        h1.remove()
        h2.remove()

        fx = fx_tensors.get("fx")
        fx_plus_x = out_tensors.get("out")

        if fx is None or fx_plus_x is None:
            return {}

        fx_mean = fx.mean(dim=1, keepdim=True)[0,0].cpu().numpy()
        fx_plus_x_mean = fx_plus_x.mean(dim=1, keepdim=True)[0,0].cpu().numpy()

        fx_mean_norm = ((fx_mean - fx_mean.min()) / (fx_mean.max() - fx_mean.min() + 1e-8) * 255).astype(np.uint8)
        fx_px_mean_norm = ((fx_plus_x_mean - fx_plus_x_mean.min()) / (fx_plus_x_mean.max() - fx_plus_x_mean.min() + 1e-8) * 255).astype(np.uint8)
        
        delta = fx_plus_x_mean - fx_mean
        delta_norm = np.zeros((*delta.shape, 3), dtype=np.uint8)
        delta_norm[delta > 0.05] = [255, 100, 100]  # Tăng cường -> Xanh lam (BGR)
        delta_norm[delta < -0.05] = [100, 100, 255] # Suy giảm -> Đỏ (BGR)
        delta_norm[(delta >= -0.05) & (delta <= 0.05)] = [255, 255, 255]

        _, fx_buf = cv2.imencode(".png", cv2.applyColorMap(fx_mean_norm, cv2.COLORMAP_VIRIDIS))
        _, fx_px_buf = cv2.imencode(".png", cv2.applyColorMap(fx_px_mean_norm, cv2.COLORMAP_VIRIDIS))
        _, delta_buf = cv2.imencode(".png", delta_norm)

        return {
            "fx": base64.b64encode(fx_buf.tobytes()).decode("utf-8"),
            "fx_plus_x": base64.b64encode(fx_px_buf.tobytes()).decode("utf-8"),
            "delta": base64.b64encode(delta_buf.tobytes()).decode("utf-8"),
            "fx_norm": float(fx.norm().item()),
            "x_norm": float((fx_plus_x - fx).norm().item()), 
            "output_norm": float(fx_plus_x.norm().item())
        }
        
    # ----- Tính toán Receptive field xấp xỉ -------------------
    def get_receptive_field(self, layer_name: str, x: int, y: int) -> dict:
        rf_size = 7
        if "stage1" in layer_name: rf_size = 35
        elif "stage2" in layer_name: rf_size = 91
        elif "stage3" in layer_name: rf_size = 155
        elif "stage4" in layer_name: rf_size = 195
        
        return {
            "bbox": [max(0, x - rf_size//2), max(0, y - rf_size//2), min(224, x + rf_size//2), min(224, y + rf_size//2)],
            "size": rf_size
        }
