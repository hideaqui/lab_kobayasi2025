import os
import shutil

# プロジェクトルート
ROOT = "/Users/hide/lab_kobayasi2025/Resnet/resnet_pytorch"

# 移動先ディレクトリ
folders = [
    "dataset", "model", "ph_analysis", "archive", "used", "used6000", "data", "output", "ripser"
]
for f in folders:
    os.makedirs(os.path.join(ROOT, f), exist_ok=True)

# ファイルの分類ルール
dataset_files = ["mnist.py", "mnist6000.py", "cifar10.py", "cifar100.py"]
model_files = ["resnet.py", "resnet3.py", "resnet50.py"]
ph_files = [
    "ph_diagram.py", "ph_lifetime.py", "ph_lifevar.py", "ph_midlife.py", "ph_gap.py", "ph_regression.py",
    "old_bunnsann.py", "old_bunnsann2.py", "old_gap.py", "old_lg_mid_pers.py", "old_midlife.py", "old_persistentlife.py", "old_ダイアグラム.py"
]
main_files = ["main_output.py", "omomi_cifar10.py", "omomi_cifar100.py"]
archive_files = ["1.py", "2.py", "15-60cifar.py"]  # 必要に応じて追加

# ファイル移動関数
def move_file(filename, target_dir):
    src = os.path.join(ROOT, filename)
    dst = os.path.join(ROOT, target_dir, filename)
    if os.path.exists(src):
        shutil.move(src, dst)
        print(f"Moved {filename} -> {target_dir}/")

# メインスクリプト
if __name__ == "__main__":
    # dataset
    for f in dataset_files:
        move_file(f, "dataset")
    # model
    for f in model_files:
        move_file(f, "model")
    # ph_analysis
    for f in ph_files:
        move_file(f, "ph_analysis")
    # archive
    for f in archive_files:
        move_file(f, "archive")
    # main scripts（ルートに残す）
    for f in main_files:
        move_file(f, ".")
    print("整理完了。不要なファイルはarchive/に移動しました。")
