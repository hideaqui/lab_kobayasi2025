from __future__ import annotations

from pathlib import Path

from lr_analysis_common import run_analysis


BASE_DIR = Path("/Users/hide/lab_kobayasi2025/Resnet/resnet_pytorch/output")
TARGETS = [
    ("kaiming_normal_1022", "kaiming_normal"),
    ("kaiming_uniform_1022", "kaiming_uniform"),
    ("normal_1022", "normal"),
    ("uniform_1022", "uniform"),
    ("xavier_normal_1022", "xavier_normal"),
    ("xavier_uniform_1022", "xavier_uniform"),
]


if __name__ == "__main__":
    for folder, label in TARGETS:
        output_dir = BASE_DIR / folder
        print(f"\n=== PH regression for {label}: {output_dir} ===")
        if not output_dir.exists():
            print(f"⚠️ 出力フォルダが見つかりません: {output_dir}")
            continue
        try:
            run_analysis(str(output_dir), label)
        except FileNotFoundError as exc:
            print(f"⚠️ {label}: {exc}")
        except Exception as exc:  # pylint: disable=broad-except
            print(f"⚠️ {label}: 解析中にエラーが発生しました -> {exc!r}")
