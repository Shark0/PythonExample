import numpy as np
import torch

if __name__ == '__main__':
    num_val = 6
    vector_dim = 4
    val_vec = np.random.rand(num_val, vector_dim).astype(np.float32)
    print("val_vec: ", val_vec)
    tensor = torch.tensor(val_vec, dtype=torch.float32)
    print("tensor: ", tensor)
    print("tensor dim: ", tensor.dim())
    print("tensor size: ", tensor.size())
    tensor = tensor.unsqueeze(0)
    print("tensor after unsqueeze: ", tensor)
    print("tensor after unsqueeze dim: ", tensor.dim())
    print("tensor after unsqueeze size: ", tensor.size())
    tensor = tensor.squeeze(0)
    print("tensor after squeeze: ", tensor)
    print("tensor after squeeze dim: ", tensor.dim())
    print("tensor after squeeze size: ", tensor.size())
    tensor = tensor.squeeze(0)
    print("tensor after squeeze again: ", tensor)
    print("tensor after squeeze again dim: ", tensor.dim())
    print("tensor after squeeze again size: ", tensor.size())