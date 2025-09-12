import numpy as np

# ファイルパスを指定
file_path = "/Users/hide/卒業研究/resnet_pytorch/output_6000/20250126_021023/epoch_1_layer4.npy"

# NumPyファイルを読み込む
layer4_activations = np.load(file_path)

# データの形状を表示
print(f"Shape of layer4 activations: {layer4_activations.shape}")

# データの内容を表示（最初の数個の要素）
print(f"First few elements of layer4 activations: {layer4_activations.flatten()[:10]}")