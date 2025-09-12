import numpy as np
from ripser import ripser
from persim import plot_diagrams

# 中間層のデータをロード
data = np.load("output/20250115_204723/epoch_1_batch_1_intermediate.npy")

# 形状を確認
print("Before reshape:", data.shape)  # (64, 512, 1, 1)

# Ripserが扱える形に変換 (64, 512)
reshaped_data = data.reshape(64, 512)
print("After reshape:", reshaped_data.shape)  # (64, 512)

# Ripserで解析
diagrams = ripser(reshaped_data,  maxdim=3)['dgms']

# 可視化
plot_diagrams(diagrams, show=True)