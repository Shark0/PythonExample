import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from recommendation_dataset import RecommendationDataset
from transformer_recommender import TransformerRecommender

def generate_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device

def generate_model(device):
    model = TransformerRecommender(
        num_categories=10, num_article_tags=20, embed_dim=64, num_heads=4, num_layers=2
    ).to(device)
    return model

def train_model(device, model, dataloader, epochs=10, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            user_seq = batch["user_category"].to(device)
            article_seq = batch["article_topic"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            scores = model(user_seq, article_seq).squeeze()
            loss = criterion(scores, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}")
    model.eval()

def recommend_articles(device, model, user_category, article_topics, top_n=5):
    user_category = torch.tensor([user_category], dtype=torch.long).to(device)
    scores = []

    for article_seq in article_topics:
        article_seq = torch.tensor([article_seq], dtype=torch.long).to(device)
        with torch.no_grad():
            score = model(user_category, article_seq).item()
        scores.append(score)

    # 排序並選 Top-N
    top_indices = torch.argsort(torch.tensor(scores), descending=True)[:top_n]
    return top_indices.tolist()

def main():
    #user_category:["toy", "food", "pet"]
    #article_topic:["toy", "food", "pet", travel]

    data = [
        {"user_category": [1, 1, 0], "article_topic": [1, 0, 0, 0], "label": 1},
        {"user_category": [1, 1, 0], "article_topic": [0, 1, 0, 1], "label": 1},
        {"user_category": [1, 1, 0], "article_topic": [0, 0, 0, 1], "label": 1},
        {"user_category": [1, 1, 0], "article_topic": [0, 0, 1, 0], "label": 0},
        {"user_category": [0, 1, 1], "article_topic": [0, 0, 1, 0], "label": 1},
        {"user_category": [0, 1, 1], "article_topic": [0, 1, 1, 0], "label": 1},
        {"user_category": [0, 1, 1], "article_topic": [0, 1, 0, 1], "label": 1},
        {"user_category": [0, 1, 1], "article_topic": [1, 0, 0, 0], "label": 0}
    ]
    dataset = RecommendationDataset(data)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    device = generate_device()
    model = generate_model(device)
    train_model(device, model, dataloader)
    user_category = [0, 1, 0]
    article_topics = [[0, 0, 1, 0], [0, 1, 1, 0], [0, 1, 0, 1], [1, 0, 0, 0], [0, 0, 1, 0]]
    top_articles = recommend_articles(device, model, user_category, article_topics, top_n=2)
    print(f"Recommended article indices: {top_articles}")

if __name__ == "__main__":
    main()
