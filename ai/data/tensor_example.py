import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader


class ExampleDataset(Dataset):
    def __init__(self, val1_metrix, val2_metrix):
        self.val1_metrix = val1_metrix
        self.val2_metrix = val2_metrix

    def __len__(self):
        return len(self.val1_metrix)

    def __getitem__(self, idx):
        val1_vec = self.val1_metrix[idx]
        val2_vec = self.val2_metrix[idx]

        return (
            torch.tensor(val1_vec, dtype=torch.float32),
            torch.tensor(val2_vec, dtype=torch.float32),
        )

def run_dataloader(dataloader):
    epochs = 2
    for epoch in range(epochs):
        print("epoch: ", epoch)
        for val1_vec, val2_vec in dataloader:
            print("val1_vec: ", val1_vec)
            print("val2_vec: ", val2_vec)


if __name__ == '__main__':
    num_value1 = 10
    num_value2 = 10
    vector_dim = 4
    val1_matrix = np.random.rand(num_value1, vector_dim).astype(np.float32)
    print("val1_matrix: ", val1_matrix)
    val2_matrix = np.random.rand(num_value2, vector_dim).astype(np.float32)
    print("val2_matrix: ", val2_matrix)
    dataset = ExampleDataset(val1_matrix, val2_matrix)
    print("datasets: ", dataset)
    dataloader = DataLoader(dataset, batch_size=5, shuffle=True)
    run_dataloader(dataloader)


