import torch
import torch.nn as nn
import torch.optim as optim

# 2. 定義簡單的線性回歸模型
class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1)  # 一個輸入，一個輸出

    def forward(self, x):
        return self.linear(x)

if __name__ == '__main__':
    torch.manual_seed(42)

    # 1. 生成模擬數據
    # 假設我們要擬合一個線性模型 y = 2x + 3 + 噪聲
    X = torch.linspace(0, 10, 100).reshape(-1, 1)  # 輸入數據
    y = 2 * X + 3 + torch.randn(X.size()) * 0.5  # 真實輸出（加噪聲）
    model = LinearRegression()
    criterion = nn.MSELoss()  # 均方誤差損失函數
    optimizer = optim.Adam(model.parameters(), lr=0.01)  # Adam最佳化器，學習率設為0.01
    num_epochs = 1000
    for epoch in range(num_epochs):
        # 前向傳播
        outputs = model(X)
        loss = criterion(outputs, y)

        # 反向傳播與最佳化
        optimizer.zero_grad()  # 清空梯度
        loss.backward()  # 計算梯度
        optimizer.step()  # 使用Adam更新參數

        # 每100個epoch打印一次損失
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    # 5. 打印最終參數
    for name, param in model.named_parameters():
        print(f'{name}: {param.data.numpy().flatten()}')

    # 6. 預測並展示結果
    with torch.no_grad():
        y_pred = model(X)
        print("\n前幾個預測值 vs 真實值:")
        for i in range(5):
            print(f'X: {X[i].item():.2f}, Predicted: {y_pred[i].item():.2f}, Actual: {y[i].item():.2f}')