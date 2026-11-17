import os
import pandas as pd
import mlflow
from mlflow import MlflowClient

# -----------------------------------------
# 1. Configuration
# -----------------------------------------

TRACKING_URI = "http://127.0.0.1:5000"

REGISTERED_MODEL_NAME = "sentiment-model"

INPUT_FILE = "data/new_data.csv"

OUTPUT_FILE = "outputs/predictions/new_data_predictions.csv"

# -----------------------------------------
# 2. Connect to MLflow
# -----------------------------------------

mlflow.set_tracking_uri(TRACKING_URI)

client = MlflowClient(tracking_uri=TRACKING_URI)

print("Connected to MLflow successfully!")

# -----------------------------------------
# 3. Get Champion model information
# -----------------------------------------

champion = client.get_model_version_by_alias(
    REGISTERED_MODEL_NAME,
    "champion"
)

model_version = champion.version

hf_model_id = champion.tags["hf_model_id"]

print("\nChampion Model Information")
print("-------------------------")
print("Registered Model:", REGISTERED_MODEL_NAME)
print("Model Version:", model_version)
print("Hugging Face Model ID:", hf_model_id)

# -----------------------------------------
# 4. Load Champion model from MLflow
# -----------------------------------------

model_uri = f"models:/{REGISTERED_MODEL_NAME}@champion"

print("\nLoading Champion model from MLflow...")

model = mlflow.pyfunc.load_model(model_uri)

print("Champion model loaded successfully!")

# -----------------------------------------
# 5. Load new input examples
# -----------------------------------------

new_data = pd.read_csv(INPUT_FILE)

if "text" not in new_data.columns:
    raise ValueError("Input CSV must contain a 'text' column.")

if new_data["text"].isnull().any():
    raise ValueError("Input data contains missing text values.")

print("\nNew examples loaded:", len(new_data))

# -----------------------------------------
# 6. Generate predictions
# -----------------------------------------

print("\nGenerating predictions...")

predictions = model.predict(new_data[["text"]])

# -----------------------------------------
# 7. Prepare prediction results
# -----------------------------------------

results = pd.DataFrame({
    "text": predictions["text"],
    "predicted_label": predictions["prediction"],
    "confidence": predictions["confidence"]
})

results["sentiment"] = results["predicted_label"].map({
    0: "NEGATIVE",
    1: "POSITIVE"
})

results["model_version"] = model_version
results["hf_model_id"] = hf_model_id

# -----------------------------------------
# 8. Display predictions
# -----------------------------------------

print("\nCHAMPION MODEL PREDICTIONS")
print("==========================")

for index, row in results.iterrows():
    print(f"\nExample {index + 1}")
    print("Text:", row["text"])
    print("Sentiment:", row["sentiment"])
    print("Confidence:", round(row["confidence"], 4))

# -----------------------------------------
# 9. Save predictions
# -----------------------------------------

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

results.to_csv(OUTPUT_FILE, index=False)

print("\n================================")
print("PREDICTIONS COMPLETED!")
print("================================")
print("Model Version:", model_version)
print("HF Model ID:", hf_model_id)
print("Total Predictions:", len(results))
print("Saved File:", OUTPUT_FILE)
