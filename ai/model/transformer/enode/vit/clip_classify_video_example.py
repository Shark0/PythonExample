import cv2
import torch
from PIL import Image
import numpy as np
from transformers import CLIPProcessor, CLIPModel

# 1. Set up CLIP model (for visual feature extraction)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


# 2. Extract video frames
def extract_frames(video_path, interval=2):
    """Extract one frame every 'interval' seconds."""
    video = cv2.VideoCapture(video_path)
    frames = []
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    while True:
        ret, frame = video.read()
        if not ret:
            break
        if frame_count % int(fps * interval) == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))
        frame_count += 1
    video.release()
    return frames


# 3. Analyze frames with CLIP to infer topics
def analyze_frames(frames):
    """Use CLIP to analyze video frames and infer topics."""
    candidate_topics = ["旅行", "美食", "運動", "科技", "教育", "娛樂", "動物", "卡通", "花", "兔子"]
    text_inputs = [f"這是一個關於{c}的影片" for c in candidate_topics]

    # Process video frames
    image_features = []
    for frame in frames:
        inputs = processor(images=frame, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            image_feature = model.get_image_features(**inputs)
        image_features.append(image_feature.cpu().numpy())

    # Average visual features
    image_features = np.mean(image_features, axis=0)
    image_features = torch.tensor(image_features).to(device)

    # Process text candidates
    text_inputs = processor(text=text_inputs, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_features = model.get_text_features(**text_inputs)

    # Compute similarity
    similarities = (image_features @ text_features.T).softmax(dim=-1)
    top_topic_idx = similarities.argmax().item()
    return candidate_topics[top_topic_idx], similarities.cpu().numpy()


# 4. Main function
def main(video_path):
    # Extract frames
    print("Extracting video frames...")
    frames = extract_frames(video_path, interval=2)  # One frame every 2 seconds

    # Analyze visual topics
    print("Analyzing visual topics...")
    if not frames:
        print("Error: No frames extracted from the video.")
        return
    visual_topic, visual_probs = analyze_frames(frames)

    # Output results
    print("\nResults:")
    print(f"Inferred topic: {visual_topic}")
    print(f"Topic probabilities: {visual_probs}")


# Execute
if __name__ == "__main__":
    video_path = "example.mp4"  # Replace with your video path
    main(video_path)