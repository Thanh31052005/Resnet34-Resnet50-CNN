import os
# Chống phân mảnh VRAM
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import time
import io
import base64
import torch
import torchvision.utils as vutils
from fastapi import APIRouter, File, UploadFile
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from torchvision.models import resnet34, resnet50, ResNet34_Weights, ResNet50_Weights

router = APIRouter()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model34 = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1).to(device).eval()
model50 = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).to(device).eval()
categories = ResNet50_Weights.IMAGENET1K_V1.meta["categories"]

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def get_top5_and_journey(model, input_batch):
    activations = {}
    stage_times = {}
    stage_starts = {}

    def get_activation(name):
        def hook(model, input, output):
            # Giải phóng VRAM
            activations[name] = output.detach().cpu()
        return hook

    def time_pre_hook(name):
        def hook(m, i):
            if device.type == 'cuda': torch.cuda.synchronize()
            stage_starts[name] = time.time()
        return hook

    def time_post_hook(name):
        def hook(m, i, o):
            if device.type == 'cuda': torch.cuda.synchronize()
            stage_times[name] = time.time() - stage_starts[name]
        return hook

    hooks = [
        model.conv1.register_forward_hook(get_activation('01_Cửa ngõ: Conv 7x7')),
        model.maxpool.register_forward_hook(get_activation('02_Cửa ngõ: MaxPool')),
        model.layer1[0].register_forward_hook(get_activation('03_Stage 1: Khối Đầu')),
        model.layer1[-1].register_forward_hook(get_activation('04_Stage 1: Khối Cuối')),
        model.layer2[0].register_forward_hook(get_activation('05_Stage 2: Khối Đầu')),
        model.layer2[-1].register_forward_hook(get_activation('06_Stage 2: Khối Cuối')),
        model.layer3[0].register_forward_hook(get_activation('07_Stage 3: Khối Đầu')),
        model.layer3[-1].register_forward_hook(get_activation('08_Stage 3: Khối Cuối')),
        model.layer4[0].register_forward_hook(get_activation('09_Stage 4: Khối Đầu')),
        model.layer4[-1].register_forward_hook(get_activation('10_Stage 4: Khối Cuối')),
        model.avgpool.register_forward_hook(get_activation('11_Lớp nén: AvgPool (Vector)')),
    ]

    stages = [
        ('Stage 1', model.layer1), ('Stage 2', model.layer2),
        ('Stage 3', model.layer3), ('Stage 4', model.layer4)
    ]
    time_hooks = []
    for stage_name, module in stages:
        time_hooks.append(module.register_forward_pre_hook(time_pre_hook(stage_name)))
        time_hooks.append(module.register_forward_hook(time_post_hook(stage_name)))

    start_total = time.time()
    with torch.no_grad():
        output = model(input_batch)
    if device.type == 'cuda': torch.cuda.synchronize()
    total_time = time.time() - start_total

    probs = torch.nn.functional.softmax(output[0], dim=0)
    top5_prob, top5_id = torch.topk(probs, 5)
    res_data = [{"label": categories[top5_id[i].item()], "conf": f"{top5_prob[i].item()*100:.2f}%"} for i in range(5)]

    for h in hooks + time_hooks: h.remove()

    journey_data = []
    for name in sorted(activations.keys()):
        tensor = activations[name]
        images_base64 = []
        
        if 'AvgPool' in name:
            channels = tensor.shape[1]
            rows = 16 if channels == 512 else 32
            cols = 32 if channels == 512 else 64
            vec = tensor[0].view(1, rows, cols)
            vec = vec - vec.min()
            vec = vec / (vec.max() + 1e-5)
            ndarr = vec.mul(255).byte().permute(1, 2, 0).repeat(1, 1, 3).numpy()
            im = Image.fromarray(ndarr)
            buffered = io.BytesIO()
            im.resize((500, 250), Image.NEAREST).save(buffered, format="PNG")
            images_base64.append("data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode("utf-8"))
            desc = f"Vector 1D ({channels}). Lưới {rows}x{cols}."
        else:
            total_channels_to_show = min(16, tensor.shape[1])
            for i in range(0, total_channels_to_show, 4):
                features = tensor[0, i:i+4, :, :].unsqueeze(1) 
                features = features - features.min()
                features = features / (features.max() + 1e-5)
                grid = vutils.make_grid(features, nrow=4, padding=4, normalize=False, pad_value=1.0)
                ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to(torch.uint8).numpy()
                im = Image.fromarray(ndarr)
                buffered = io.BytesIO()
                im.save(buffered, format="PNG")
                images_base64.append("data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode("utf-8"))
            desc = f"Shape: {tensor.shape[2]}x{tensor.shape[3]} | Tổng Channels: {tensor.shape[1]}"
        
        journey_data.append({
            "step": name.split('_', 1)[1],
            "desc": desc,
            "images": images_base64
        })

    return res_data, f"{total_time:.4f}s", stage_times, journey_data

@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    
    input_batch = preprocess(img).unsqueeze(0).to(device)

    # Quá trình Inference
    res34, time34, stages34, journey_34 = get_top5_and_journey(model34, input_batch)
    res50, time50, stages50, journey_50 = get_top5_and_journey(model50, input_batch)
    
    # Dọn rác VRAM
    del input_batch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "device": str(device),
        "resnet34": {"data": res34, "total_time": time34, "stage_times": stages34, "journey": journey_34},
        "resnet50": {"data": res50, "total_time": time50, "stage_times": stages50, "journey": journey_50}
    }