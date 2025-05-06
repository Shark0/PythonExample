import torch
import torch.nn as nn

if __name__ == '__main__':
    torch.manual_seed(42)
    labels = torch.tensor([1.0, 0.0, 1.0, 1.0])
    logits = torch.tensor([2.0, -1.5, 1.5, 0.5], requires_grad=True)  # 需要計算梯度

    criterion = nn.CrossEntropyLoss()
    loss = criterion(logits, labels)
    print(f"Binary Cross-Entropy Loss: {loss.item():.4f}")
    loss.backward()
    print(f"Gradients of logits: {logits.grad}")

    # 模擬優化器更新（假設學習率為 0.1）
    learning_rate = 0.1
    with torch.no_grad():
        logits -= learning_rate * logits.grad
        logits.grad.zero_()

    print(f"Updated logits: {logits}")