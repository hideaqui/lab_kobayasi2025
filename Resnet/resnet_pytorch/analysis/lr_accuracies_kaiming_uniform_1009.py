from lr_analysis_common import run_analysis

OUTPUT_DIR = "/Users/hide/lab_kobayasi2025/output/kaiming_uniform_1009"
INIT_LABEL = "kaiming_uniform"


if __name__ == "__main__":
    try:
        run_analysis(OUTPUT_DIR, INIT_LABEL)
    except FileNotFoundError as error:
        print(f"[kaiming_uniform] 分析をスキップします: {error}")
