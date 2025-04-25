from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image
import requests
import torch
import matplotlib.pyplot as plt

# Load a sample image
url = 'https://pngimg.com/d/shark_PNG18830.png'
image = Image.open(requests.get(url, stream=True).raw)
image = image.convert("RGB")
# Load feature extractor and model
extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k')

# Preprocess the image
inputs = extractor(images=image, return_tensors="pt")

# Forward pass
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

# Get prediction
predicted_class = logits.argmax(-1).item()
label = model.config.id2label[predicted_class]

# Show result
plt.imshow(image)
plt.title(f"Predicted Label: {label}")
plt.axis("off")
plt.show()