import numpy as np
import kmapper as km
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

# 1) numpyファイルの読み込み
file_path = "/Users/hide/卒業研究/resnet_pytorch/output_6000/20250126_021306/epoch_200_layer4.npy"
activation_data = np.load(file_path)

# 2) reshape: (N, C, 1, 1) → (N, C)
activation_data_reshaped = activation_data.reshape(activation_data.shape[0], -1)
print("activation_data_reshaped.shape:", activation_data_reshaped.shape)

# 3) KeplerMapper インスタンスの作成
mapper = km.KeplerMapper(verbose=1)

# 4) Lens (次元削減/投影関数) の選択 (PCA + StandardScaler)
lens = mapper.fit_transform(activation_data_reshaped, projection=PCA(n_components=10), scaler=StandardScaler())

# 5) Mapper の作成 (カバリング数を減らす)
graph = mapper.map(
    lens,
    activation_data_reshaped,
    cover=km.Cover(n_cubes=5, perc_overlap=0.2),  # 分割数を減らす
    clusterer=DBSCAN(eps=0.3, min_samples=5)  # DBSCAN のパラメータを調整
)

# 6) 可視化の生成 (HTML)
out_html = "mapper_visualization.html"
mapper.visualize(
    graph,
    path_html=out_html,
    title="Mapper Visualization for epoch_0_layer4"
)

print(f"✅ Mapper の可視化結果を '{out_html}' に保存しました。")