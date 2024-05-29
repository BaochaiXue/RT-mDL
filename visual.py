import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the results from CSV file
df = pd.read_csv("model_variants_results.csv")

# Plot accuracy vs. pruning amount
plt.figure(figsize=(12, 6))
sns.scatterplot(
    data=df,
    x="pruning_amount",
    y="accuracy",
    hue="width_scaling_factor",
    style="depth_scaling_factor",
    palette="viridis",
)
plt.title("Accuracy vs. Pruning Amount")
plt.xlabel("Pruning Amount")
plt.ylabel("Accuracy")
plt.legend(title="Width Scaling Factor and Depth Scaling Factor")
plt.show()

# Plot test time vs. pruning amount
plt.figure(figsize=(12, 6))
sns.scatterplot(
    data=df,
    x="pruning_amount",
    y="test_time",
    hue="width_scaling_factor",
    style="depth_scaling_factor",
    palette="viridis",
)
plt.title("Test Time vs. Pruning Amount")
plt.xlabel("Pruning Amount")
plt.ylabel("Test Time (seconds)")
plt.legend(title="Width Scaling Factor and Depth Scaling Factor")
plt.show()

# You can create similar plots for other parameters as needed
