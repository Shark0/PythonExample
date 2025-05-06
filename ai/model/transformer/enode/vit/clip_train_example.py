import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from transformers import CLIPModel, CLIPProcessor
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import torch.nn.functional as F
from PIL import Image
import numpy as np

def generate_train_dataloader():
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    train_dataset = datasets.ImageFolder(root='dataset/train', transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    return train_loader

def generate_value_dataloader():
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    val_dataset = datasets.ImageFolder(root='dataset/value', transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    return val_loader

def generate_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device

def generate_processor(model_name):
    processor = CLIPProcessor.from_pretrained(model_name)
    return processor

def train_model(model_name, processor, device, train_dataloader, val_dataloader):
    model = CLIPModel.from_pretrained(model_name)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=1e-5)
    criterion = CrossEntropyLoss()
    epochs = 3  # 小數據集建議 3-5 個 epoch
    texts = ["A photo of an animal", "A photo of a landscape"]
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}"):
            images, labels = images.to(device), labels.to(device)

            # 預處理圖片和文本
            inputs = processor(
                text=texts,
                images=images,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # 前向傳播
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image  # 圖片-文本相似度分數

            # 計算損失
            loss = criterion(logits_per_image, labels)

            # 反向傳播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        avg_loss = running_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1}, Average Loss: {avg_loss:.4f}")

    # 保存模型
    model.save_pretrained("clip_animal_landscape")
    processor.save_pretrained("clip_animal_landscape")
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in tqdm(val_dataloader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)

            # 預處理
            inputs = processor(
                text=texts,
                images=images,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # 推理
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            predicted = logits_per_image.argmax(dim=1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total * 100
    print(f"Validation Accuracy: {accuracy:.2f}%")
    return model

def classify_image_with_clip(image_path, model, processor, texts, device):
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error loading image: {e}")
        return None, None
    # 預處理
    inputs = processor(
        text=texts,
        images=image,
        return_tensors="pt",
        padding=True,
        truncation=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    # 推理
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = F.softmax(logits_per_image, dim=1).cpu().numpy()[0]

    # 輸出結果
    class_names = ['Animal', 'Landscape']
    predicted_class = np.argmax(probs)
    print(f"Predicted Class: {class_names[predicted_class]}")
    print(f"Confidence Scores: Animal: {probs[0]:.4f}, Landscape: {probs[1]:.4f}")
    return predicted_class, probs


def main():
    train_data_loader = generate_train_dataloader()
    value_dataloader = generate_value_dataloader()
    device = generate_device()
    model_name = "openai/clip-vit-base-patch32"
    processor = generate_processor(model_name)
    model = train_model(model_name, processor, device, train_data_loader, value_dataloader)
    image_path = "dataset/test/test.jpg"  # 替換為你的測試圖片路徑
    texts = ["A photo of an animal", "A photo of a landscape"]
    classify_image_with_clip(image_path, model, processor, texts, device)


if __name__ == "__main__":
    main()
