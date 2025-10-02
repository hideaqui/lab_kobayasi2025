import os
import numpy as np
import matplotlib.pyplot as plt
from ripser import ripser
from persim import plot_diagrams

# データ保存フォルダ
output_dir = "/Users/hide/lab_kobayasi2025/output/uniform_20251001_004321"

# **解析するエポック数**
num_epochs = 21  

# **結果を保存するリスト**
file_names = []
persistence_diagrams = []
persistence_pairs = []  # (b, d) のリスト

# **エポックごとの Layer4 活性化データを読み込み、パーシステントホモロジーを計算**
for epoch in range(num_epochs):
    file_path = os.path.join(output_dir, f"epoch_{epoch}_layer4.npy")
    file_names.append(file_path)

    if os.path.exists(file_path):
        activation_data = np.load(file_path)  # Layer4 の活性化データ読み込み

        # **データを 2次元に変換 (batch_size, feature_dim)**
        activation_data = activation_data.reshape(activation_data.shape[0], -1)

        # **Ripser でパーシステントホモロジー計算**
        diagrams = ripser(activation_data.T, maxdim=2)['dgms']
        persistence_diagrams.append(diagrams)

        # **(b, d) ペアを取得**
        epoch_persistence_pairs = []
        for dim, diag in enumerate(diagrams):  # H0, H1, H2 のそれぞれのペア
            for point in diag:
                epoch_persistence_pairs.append((point[0], point[1]))  # (birth, death)
        
        persistence_pairs.append(epoch_persistence_pairs)

    else:
        print(f"Warning: {file_path} が見つかりません")
        persistence_diagrams.append(None)
        persistence_pairs.append(None)

# **パーシステント図を分割表示**
fig, axes = plt.subplots(5, 5, figsize=(20, 16))
fig.suptitle("Persistent Diagrams for Epoch 0-20 (Layer4 Activations)")

for epoch in range(num_epochs):
    ax = axes[epoch // 5, epoch % 5]
    
    if persistence_diagrams[epoch] is not None:
        plot_diagrams(persistence_diagrams[epoch], ax=ax)
        ax.set_title(f"Epoch {epoch}")
    else:
        ax.axis('off')
        ax.set_title(f"Epoch {epoch} (No Data)")

plt.tight_layout()
plt.show()

# **ファイル名リスト表示**
print("### Processed Files ###")
for f in file_names:
    print(f)

# **パーシステンスペアリスト表示**
print("\n### Persistence Pairs (b, d) ###")
for epoch in range(num_epochs):
    print(f"Epoch {epoch}:")
    if persistence_pairs[epoch] is not None:
        print(persistence_pairs[epoch])
    else:
        print("No Data")