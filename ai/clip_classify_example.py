import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests

# 設置設備
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 加載CLIP模型和處理器
try:
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.to(device)
    model.eval()
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

def classify_image_with_clip(image_url):
    try:
        # 加載圖片（添加User-Agent以避免訪問限制）
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(image_url, stream=True, headers=headers, timeout=10)
        response.raise_for_status()  # 檢查請求是否成功
        image = Image.open(response.raw).convert("RGB")
    except Exception as e:
        print(f"Error loading image: {e}")
        return None, None

    # 預處理圖片和文本
    texts = ["A photo of an animal", "A photo of a landscape"]
    try:
        inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
    except Exception as e:
        print(f"Error processing inputs: {e}")
        return None, None

    # 推理
    try:
        with torch.no_grad():
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1).cpu().numpy()[0]
    except Exception as e:
        print(f"Error during inference: {e}")
        return None, None

    # 輸出結果
    class_names = ['Animal', 'Landscape']
    predicted_class = np.argmax(probs)
    print(f"Predicted Class: {class_names[predicted_class]}")
    print(f"Confidence Scores: Animal: {probs[0]:.4f}, Landscape: {probs[1]:.4f}")
    return predicted_class, probs

# 示例
image_url = 'https://today-obs.line-scdn.net/0hJwFUI87XFVh7LAOPMLJqD0F6FjdIQAZbHxpEWzhCS29XT1MLQBhSblh4GW5VFFIGFR1YOlosDmkDSAAJEhlS/w1200'
classify_image_with_clip(image_url)