import torch

def intense_computation():
    # 确保使用GPU进行计算
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 创建一个小的随机矩阵
    data = torch.randn(100, 100, device=device)

    # 执行大量的计算操作，但不增加显存占用
    while True:
        data = data * torch.randn(100, 100, device=device)

    # 输出最终的计算结果（仅为示例，实际可能不需要）
    print(data)

if __name__ == "__main__":
    intense_computation()