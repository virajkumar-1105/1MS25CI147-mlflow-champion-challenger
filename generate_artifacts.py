
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# -----------------------------------------
# 1. File paths
# -----------------------------------------

MODEL_A_FILE = "outputs/predictions/A_predictions.csv"
MODEL_B_FILE = "outputs/predictions/B_predictions.csv"

PLOT_DIR = "outputs/plots"

os.makedirs(PLOT_DIR, exist_ok=True)

# -----------------------------------------
# 2. Load prediction results
# -----------------------------------------

model_a = pd.read_csv(MODEL_A_FILE)
model_b = pd.read_csv(MODEL_B_FILE)

print("Model A predictions loaded:", len(model_a))
print("Model B predictions loaded:", len(model_b))

# Verify that both models used the same examples
if not model_a["text"].equals(model_b["text"]):
    raise ValueError("Model A and B inputs do not match!")

if not model_a["actual_label"].equals(model_b["actual_label"]):
    raise ValueError("Actual labels do not match!")

y_true = model_a["actual_label"]

# -----------------------------------------
# 3. Generate Model A confusion matrix
# -----------------------------------------

cm_a = confusion_matrix(
    model_a["actual_label"],
    model_a["predicted_label"],
    labels=[0, 1]
)

disp_a = ConfusionMatrixDisplay(
    confusion_matrix=cm_a,
    display_labels=["Negative", "Positive"]
)

disp_a.plot(cmap="Blues", values_format="d")
plt.title("Model A - DistilBERT Confusion Matrix")
plt.tight_layout()

plt.savefig(
    f"{PLOT_DIR}/model_a_confusion_matrix.png",
    dpi=300
)

plt.close()

# -----------------------------------------
# 4. Generate Model B confusion matrix
# -----------------------------------------

cm_b = confusion_matrix(
    model_b["actual_label"],
    model_b["predicted_label"],
    labels=[0, 1]
)

disp_b = ConfusionMatrixDisplay(
    confusion_matrix=cm_b,
    display_labels=["Negative", "Positive"]
)

disp_b.plot(cmap="Greens", values_format="d")
plt.title("Model B - BERT Confusion Matrix")
plt.tight_layout()

plt.savefig(
    f"{PLOT_DIR}/model_b_confusion_matrix.png",
    dpi=300
)

plt.close()

# -----------------------------------------
# 5. Generate model comparison chart
# -----------------------------------------

metrics = ["accuracy", "f1_macro"]

values_a = [
    (model_a["predicted_label"] == model_a["actual_label"]).mean(),
    0.9141
]

values_b = [
    (model_b["predicted_label"] == model_b["actual_label"]).mean(),
    0.9497
]

x = range(len(metrics))
width = 0.35

plt.figure(figsize=(8, 5))

plt.bar(
    [i - width / 2 for i in x],
    values_a,
    width,
    label="Model A - DistilBERT"
)

plt.bar(
    [i + width / 2 for i in x],
    values_b,
    width,
    label="Model B - BERT"
)

plt.xticks(list(x), ["Accuracy", "Macro F1"])
plt.ylim(0, 1.05)
plt.ylabel("Score")
plt.title("Champion-Challenger Model Comparison")
plt.legend()
plt.tight_layout()

plt.savefig(
    f"{PLOT_DIR}/model_comparison.png",
    dpi=300
)

plt.close()

# -----------------------------------------
# 6. Find disagreement examples
# -----------------------------------------

comparison = pd.DataFrame({
    "text": model_a["text"],
    "actual_label": model_a["actual_label"],
    "model_a_prediction": model_a["predicted_label"],
    "model_b_prediction": model_b["predicted_label"]
})

disagreements = comparison[
    comparison["model_a_prediction"] !=
    comparison["model_b_prediction"]
].copy()

if len(disagreements) < 5:
    raise ValueError(
        f"Only {len(disagreements)} disagreements found. "
        "At least 5 are required."
    )

disagreements = disagreements.head(5)

# Add an explanation for each disagreement
def explain_disagreement(row):
    actual = "POSITIVE" if row["actual_label"] == 1 else "NEGATIVE"

    pred_a = (
        "POSITIVE" if row["model_a_prediction"] == 1
        else "NEGATIVE"
    )

    pred_b = (
        "POSITIVE" if row["model_b_prediction"] == 1
        else "NEGATIVE"
    )

    correct_a = row["model_a_prediction"] == row["actual_label"]
    correct_b = row["model_b_prediction"] == row["actual_label"]

    if correct_a and not correct_b:
        reason = "Model A matched the actual label; Model B did not."
    elif correct_b and not correct_a:
        reason = "Model B matched the actual label; Model A did not."
    else:
        reason = "Both models disagreed with each other; neither prediction matched the actual label."

    return (
        f"Actual: {actual}. "
        f"Model A: {pred_a}. Model B: {pred_b}. "
        f"{reason}"
    )

disagreements["comment"] = disagreements.apply(
    explain_disagreement,
    axis=1
)

disagreement_file = "outputs/disagreement_examples.csv"

disagreements.to_csv(disagreement_file, index=False)

# -----------------------------------------
# 7. Display completion information
# -----------------------------------------

print("\nARTIFACT GENERATION COMPLETED!")
print("--------------------------------")
print("Plots saved in:", PLOT_DIR)
print("Model A confusion matrix saved.")
print("Model B confusion matrix saved.")
print("Model comparison chart saved.")
print("Disagreement examples saved:", disagreement_file)
print("\nNumber of disagreements:", len(comparison[
    comparison["model_a_prediction"] !=
    comparison["model_b_prediction"]
]))
