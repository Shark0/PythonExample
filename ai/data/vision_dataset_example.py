import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torchvision.transforms as transforms

def load_dataset():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    return train_dataset, test_dataset

def visualize_dataset(dataset, title, num_images=9):
    # 創建一個 3x3 網格
    fig, axes = plt.subplots(3, 3, figsize=(8, 8))
    fig.suptitle(title, fontsize=16)

    # 隨機選擇 num_images 張圖像
    indices = np.random.choice(len(dataset), num_images, replace=False)

    for i, idx in enumerate(indices):
        # 獲取圖像和標籤
        img, label = dataset[idx]
        # 將張量轉為 numpy 並反標準化
        img = img.numpy().squeeze()  # 移除通道維度 (1, 28, 28) -> (28, 28)
        img = img * 0.5 + 0.5  # 反標準化：從 [-1, 1] 轉回 [0, 1]

        # 繪製圖像
        ax = axes[i // 3, i % 3]
        ax.imshow(img, cmap='gray')
        ax.set_title(f'Label: {label}')
        ax.axis('off')  # 隱藏坐標軸

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # 調整佈局以容納標題
    plt.show()

if __name__ == '__main__':
    train_dataset, test_dataset = load_dataset()
    visualize_dataset(train_dataset, 'MNIST Training Dataset Samples')
    visualize_dataset(test_dataset, 'MNIST Test Dataset Samples')