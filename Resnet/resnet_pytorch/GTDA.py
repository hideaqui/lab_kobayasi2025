import numpy as np
import matplotlib.pyplot as plt
from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import PersistenceLandscape
from sklearn.preprocessing import StandardScaler

# データのロード
file_path = "/Users/hide/卒業研究/resnet_pytorch/output_6000/20250126_021306/epoch_200_layer4.npy"
data = np.load(file_path)

# データの形状を確認
print(f"Original shape: {data.shape}")

# 4次元データを2次元に変換
if data.ndim == 4:
    data = data.reshape(data.shape[0], -1)  # (サンプル数, 特徴量数) に変換
elif data.ndim > 2:
    raise ValueError(f"Unexpected data shape {data.shape}. Ensure it's (samples, features).")

print(f"Reshaped shape: {data.shape}")

# データの標準化
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# パーシステントホモロジーの計算
VR_persistence = VietorisRipsPersistence(metric="euclidean", homology_dimensions=[0, 1, 2])
diagrams = VR_persistence.fit_transform(data_scaled[None, :, :])  # 3次元 (1, サンプル数, 特徴量数) にする

# パーシステンスランドスケープの計算
pl = PersistenceLandscape(n_layers=5)
landscapes = pl.fit_transform(diagrams)

# ランドスケープの可視化
plt.figure(figsize=(10, 6))
for i in range(5):  # 上位5つのランドスケープを描画
    plt.plot(landscapes[0, :, i], label=f"Layer {i+1}")
plt.xlabel("Filtration value")
plt.ylabel("Persistence Landscape")
plt.title("Persistence Landscape for ResNet Layer4 Activations")
plt.legend()
plt.show()