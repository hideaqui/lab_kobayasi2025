# used6000/ の実験・特殊用途スクリプトは archive6000/ フォルダに整理

# 推奨ディレクトリ構成

Resnet/
  resnet_pytorch/
    main_output.py                # MNIST/CIFAR-10学習・中間層保存（本番用）
    omomi_cifar10.py              # CIFAR-10学習（本番用）
    omomi_cifar100.py             # CIFAR-100学習（本番用）
    dataset/
      mnist.py                    # MNISTデータ取得
      mnist6000.py                # MNIST(6000)データ取得
      cifar10.py                  # CIFAR-10データ取得
      cifar100.py                 # CIFAR-100データ取得
    model/
      resnet.py                   # ResNet本体
      resnet3.py
      resnet50.py
    ph_analysis/                  # パーシステントホモロジー解析・可視化
      ph_diagram.py               # ダイアグラム可視化
      ph_lifetime.py              # ライフタイム平均
      ph_lifevar.py               # ライフタイム分散
      ph_midlife.py               # ミッドライフ分散
      ph_gap.py                   # 汎化ギャップ可視化
      ph_regression.py            # 汎化ギャップ回帰分析
      old_bunnsann.py             # バックアップ
      old_bunnsann2.py
      old_gap.py
      old_lg_mid_pers.py
      old_midlife.py
      old_persistentlife.py
      old_ダイアグラム.py
      README.md
    archive/                      # 一時・実験・古いスクリプト
      1.py
      2.py
      15-60cifar.py
      ...（他の古いスクリプトもここへ）
    used/                         # さらに一時的な実験コード（必要なら）
    used6000/                     # 6000バッチ用実験コード（必要なら）
    data/                         # データセット（.gitignore推奨）
      MNIST/
    output/                       # 学習・解析出力（.gitignore推奨）
    ripser/                       # トポロジー解析用サブモジュール
      poll.py
      poll2.py
      test/
        simple1.py
        simple2.py
    README.md