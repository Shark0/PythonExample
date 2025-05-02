import torch
from torch.utils.data import Dataset


class RecommendationDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "user_category": torch.tensor(item["user_category"], dtype=torch.long),
            "article_topic": torch.tensor(item["article_topic"], dtype=torch.long),
            "label": torch.tensor(item["label"], dtype=torch.float)
        }