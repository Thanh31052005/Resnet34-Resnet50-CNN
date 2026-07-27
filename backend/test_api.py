"""Quick test for /predict endpoint."""
import requests
from PIL import Image
import io

# Tao anh test 224x224
img = Image.new('RGB', (224, 224), color=(255, 100, 50))
buf = io.BytesIO()
img.save(buf, format='PNG')
buf.seek(0)

print("Sending request to /predict...")
r = requests.post(
    'http://localhost:8000/predict',
    files={'file': ('test.png', buf, 'image/png')},
    timeout=120
)
print('Status:', r.status_code)
if r.status_code == 200:
    data = r.json()
    print('Prediction:', data.get('prediction'))
    print('Confidence:', data.get('confidence'))
    print('Layers count:', len(data.get('layers', [])))
    print('Has gradcam:', bool(data.get('gradcam_overlay')))
    for layer in data.get('layers', []):
        name = layer['name']
        shape = layer['output_shape']
        n_feat = len(layer.get('feature_maps', []))
        print(f"  {name}: shape={shape}, features={n_feat}")
else:
    print('Error:', r.text[:500])
