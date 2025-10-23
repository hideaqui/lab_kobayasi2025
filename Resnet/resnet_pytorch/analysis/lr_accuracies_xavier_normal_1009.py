from lr_analysis_common import run_analysis

OUTPUT_DIR = "/Users/hide/lab_kobayasi2025/output/xavier_normal_1009"
INIT_LABEL = "xavier_normal"


if __name__ == "__main__":
    try:
        run_analysis(OUTPUT_DIR, INIT_LABEL)
    except FileNotFoundError as error:
        print(f"[xavier_normal] 分析をスキップします: {error}")
