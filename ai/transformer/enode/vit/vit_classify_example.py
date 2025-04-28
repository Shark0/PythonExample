import torch
from transformers import ViTForImageClassification, ViTFeatureExtractor
from PIL import Image
import torch.nn.functional as F
import numpy as np
import requests

# 加載預訓練模型和特徵提取器
model_name = "google/vit-base-patch16-224-in21k"
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
model = ViTForImageClassification.from_pretrained(model_name, num_labels=2, ignore_mismatched_sizes=True)

# 設置設備
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()  # 推理模式


def classify_image(image_url):
    # 打開並預處理圖片
    image = Image.open(requests.get(image_url, stream=True).raw)
    image = image.convert("RGB")
    inputs = feature_extractor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 模型推理
    with torch.no_grad():
        outputs = model(**inputs).logits
        probabilities = F.softmax(outputs, dim=1)  # 轉為概率
        predicted_class = probabilities.argmax(-1).item()
        confidence_scores = probabilities[0].cpu().numpy()

    # 類別名稱
    class_names = ['Animal', 'Landscape']

    # 輸出結果
    print(f"Predicted Class: {class_names[predicted_class]}")
    print(f"Confidence Scores: Animal: {confidence_scores[0]:.4f}, Landscape: {confidence_scores[1]:.4f}")
    return predicted_class, confidence_scores


# 示例：測試一張圖片
image_url = 'https://today-obs.line-scdn.net/0hJwFUI87XFVh7LAOPMLJqD0F6FjdIQAZbHxpEWzhCS29XT1MLQBhSblh4GW5VFFIGFR1YOlosDmkDSAAJEhlS/w1200'  # 替換為你的圖片路徑
classify_image(image_url)