import matplotlib.pyplot as plt
import pandas as pd

result_path = "/Users/hide/卒業研究/resnet_pytorch/output_6000/20250125_030030/regression_results_dimwise_1-10.csv"
df = pd.read_csv(result_path)

plt.figure(figsize=(8, 5))
plt.scatter(df["epoch"], df["actual_gap"], label="Actual Gap", marker="o", alpha=0.7)
plt.plot(df["epoch"], df["predicted_gap"], label="Predicted Gap", linestyle="--", color="red")

plt.xlabel("Epoch")
plt.ylabel("Generalization Gap")
plt.legend()
plt.title("Generalization Gap Prediction (Epoch 1-10)")
plt.grid()
plt.show()