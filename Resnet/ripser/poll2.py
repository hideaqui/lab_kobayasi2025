
import numpy as np
import kmapper as km
from sklearn import datasets
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

# データセットの生成 (ここでは合成データを使用)
data, labels = datasets.make_moons(n_samples=1000, noise=0.05)

# KeplerMapper オブジェクトの初期化
mapper = km.KeplerMapper(verbose=1)

# データにフィルタリング関数を適用 (PCAによる次元削減をフィルターとして使用)
projected_data = mapper.fit_transform(data, projection=PCA(n_components=1))

# カバーの設定とクラスタリングアルゴリズムの適用
graph = mapper.map(projected_data, data, clusterer=DBSCAN(eps=0.3, min_samples=10))

# 可視化 (ここではHTMLファイルとして出力)
mapper.visualize(graph, path_html="mapper_visualization.html", title="Mapper visualization of moon dataset")

