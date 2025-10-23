#!/bin/bash

# Conda環境を有効化して順次実行
echo "=== Running all initialization modes ==="

conda run -n da python main1022/main_kaiming_normal_1022.py
conda run -n da python main1022/main_normal_1022.py
conda run -n da python main1022/main_uniform_1022.py
conda run -n da python main1022/main_xavier_normal_1022.py
conda run -n da python main1022/main_xavier_uniform_1022.py
conda run -n da python main1022/main_kaiming_uniform_1022.py

echo "=== All runs completed successfully! ==="