# lab_kobayasi2025
ibis用code

## フォルダ構成の説明

- 大前提：全部がresnet_pytorchにはいっている。

## resnet_pytorchの説明
- datasetフォルダには、学習時にデータを取得するコードが入っている。
- modelは機械学習モデルの初期値設定をしている（ランダム行列の設定などもここ）
- ph_analysisでは中間層のデータからpd図を作成して、さらにmidpoint（各ホモロジーのbirthとdeathの中間点）と、パーシステンス（各ホモロジーのdeath - birth）から汎化誤差（テストエラー）の線形回帰をおこなう。