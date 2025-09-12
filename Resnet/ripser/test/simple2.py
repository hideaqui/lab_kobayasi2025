import numpy as np
from ripser import Rips

# Ripsオブジェクトの作成
rips = Rips()

# ランダムな2次元データを生成
data = np.random.random((100, 2))

# パーシステンス図を計算
diagrams = rips.fit_transform(data)

# パーシステンス図をプロット
rips.plot(diagrams)