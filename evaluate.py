import os
import gc
import time
import json

import pandas as pd
import mlflow
import mlflow.pyfunc

from transformers import pipeline

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec


# --------------------------------------------------
# 1. PROJECT CONFIGURATION
# --------------------------------------------------

TRACKING_URI = "http://127.0.0.1:5000"

EXPERIMENT_NAME = "hf-sentiment-champion-challenger"

DATASET_PATH = "data/eval_inputs.csv"

RANDOM_SEED = 42

BATCH_SIZE = 2

MAX_LENGTH = 256

DEVICE = -1  # -1 means CPU


# --------------------------------------------------
# 2. STUDENT INFORMATION
# --------------------------------------------------

STUDENT_USN = "1MS25CI147"


# --------------------------------------------------
# 3. MODEL CONFIGURATION
# --------------------------------------------------

MODELS = {
    "A": {
        "name": "distilbert-sst2",
        "model_id": "distilbert-base-uncased-finetuned-sst-2-english",
        "role": "champion"
    },

    "B": {
        "name": "bert-sst2",
        "model_id": "textattack/bert-base-uncased-SST-2",
        "role": "challenger"
    }
}


# --------------------------------------------------
# 4. CONNECT TO MLFLOW
# --------------------------------------------------

mlflow.set_tracking_uri(TRACKING_URI)

mlflow.set_experiment(EXPERIMENT_NAME)


print("MLflow connected successfully!")

print("Experiment:", EXPERIMENT_NAME)

print("Student USN:", STUDENT_USN)

# --------------------------------------------------
# 5. LOAD AND VALIDATE EVALUATION DATA
# --------------------------------------------------

def load_evaluation_data():

    print("\nLoading evaluation dataset...")

    df = pd.read_csv(DATASET_PATH)

    # Check required columns
    if not {"text", "label"}.issubset(df.columns):
        raise ValueError("CSV must contain 'text' and 'label' columns.")

    # Check for missing values
    if df[["text", "label"]].isnull().any().any():
        raise ValueError("Dataset contains missing values.")

    # Convert labels to integers
    df["label"] = df["label"].astype(int)

    # Check that labels are only 0 and 1
    if not set(df["label"].unique()).issubset({0, 1}):
        raise ValueError("Labels must be either 0 or 1.")

    print("Dataset loaded successfully!")
    print("Total evaluation samples:", len(df))
    print("Label distribution:")
    print(df["label"].value_counts().sort_index())

    return df


# --------------------------------------------------
# 6. CONVERT MODEL PREDICTIONS TO NUMERIC LABELS
# --------------------------------------------------

def convert_prediction(label):

    label = label.upper()

    if label in ["POSITIVE", "LABEL_1"]:
        return 1

    elif label in ["NEGATIVE", "LABEL_0"]:
        return 0

    else:
        raise ValueError(f"Unknown prediction label: {label}")

# --------------------------------------------------
# 7. CALCULATE CLASSIFICATION METRICS
# --------------------------------------------------

def calculate_metrics(actual, predicted):

    metrics = {
        "accuracy": accuracy_score(actual, predicted),

        "precision_macro": precision_score(
            actual, predicted, average="macro", zero_division=0
        ),

        "recall_macro": recall_score(
            actual, predicted, average="macro", zero_division=0
        ),

        "f1_macro": f1_score(
            actual, predicted, average="macro", zero_division=0
        )
    }

    return metrics


# --------------------------------------------------
# 8. MLFLOW PYTHON MODEL WRAPPER
# --------------------------------------------------

class HuggingFaceSentimentModel(mlflow.pyfunc.PythonModel):

    def __init__(self, model_id):
        self.model_id = model_id

    # Load the Hugging Face model when MLflow loads this model
    def load_context(self, context):

        self.classifier = pipeline(
            "sentiment-analysis",
            model=self.model_id,
            tokenizer=self.model_id,
            device=-1
        )

    # Generate predictions through MLflow
    def predict(self, context, model_input, params=None):

        texts = model_input["text"].astype(str).tolist()

        raw_predictions = self.classifier(
            texts,
            truncation=True,
            max_length=MAX_LENGTH
        )

        return pd.DataFrame({
            "text": texts,
            "prediction": [
                convert_prediction(item["label"])
                for item in raw_predictions
            ],
            "confidence": [
                item["score"] for item in raw_predictions
            ]
        })


# Define the model's input and output structure
MODEL_SIGNATURE = ModelSignature(
    inputs=Schema([
        ColSpec("string", "text")
    ]),
    outputs=Schema([
        ColSpec("string", "text"),
        ColSpec("long", "prediction"),
        ColSpec("double", "confidence")
    ])
)

def evaluate_model(model_key, model_config, df):

    model_id = model_config["model_id"]
    model_name = model_config["name"]
    role = model_config["role"]

    run_name = f"{model_key}_sentiment_{model_name}"

    print("\n" + "=" * 60)
    print("Evaluating:", run_name)
    print("Model ID:", model_id)
    print("=" * 60)

    # Extract the same evaluation inputs for both models
    texts = df["text"].tolist()
    actual_labels = df["label"].tolist()

    # Start an MLflow run
    with mlflow.start_run(run_name=run_name) as run:

        # Record model configuration
        mlflow.log_params({
            "hf_model_id": model_id,
            "task": "sentiment-classification",
            "role": role,
            "max_length": MAX_LENGTH,
            "batch_size": BATCH_SIZE,
            "eval_samples": len(df),
            "seed": RANDOM_SEED,
            "device": "cpu"
        })

        # Record student and software information
        mlflow.set_tags({
            "student_usn": STUDENT_USN,
            "dataset_name": "nyu-mll/glue-sst2",
            "transformers_version": __import__("transformers").__version__,
            "torch_version": __import__("torch").__version__,
            "mlflow_version": mlflow.__version__
        })

        # Load the Hugging Face model
        print("\nLoading model...")

        classifier = pipeline(
            "sentiment-analysis",
            model=model_id,
            tokenizer=model_id,
            device=DEVICE
        )

        print("Model loaded successfully!")

        # Measure inference time only
        start_time = time.perf_counter()

        raw_predictions = classifier(
            texts,
            batch_size=BATCH_SIZE,
            truncation=True,
            max_length=MAX_LENGTH
        )

        end_time = time.perf_counter()

        total_inference_s = end_time - start_time

        avg_latency_ms = (
            total_inference_s / len(texts)
        ) * 1000

        # Convert predictions into 0 and 1
        predicted_labels = [
            convert_prediction(item["label"])
            for item in raw_predictions
        ]

        # Calculate classification metrics
        metrics = calculate_metrics(
            actual_labels,
            predicted_labels
        )

        metrics["total_inference_s"] = total_inference_s
        metrics["avg_latency_ms"] = avg_latency_ms

        # Log metrics to MLflow
        mlflow.log_metrics(metrics)

        # Create prediction results
        results_df = pd.DataFrame({
            "text": texts,
            "actual_label": actual_labels,
            "predicted_label": predicted_labels,
            "confidence": [
                item["score"] for item in raw_predictions
            ],
            "correct": [
                actual == predicted
                for actual, predicted in zip(
                    actual_labels, predicted_labels
                )
            ]
        })

        # Save prediction CSV
        os.makedirs("outputs/predictions", exist_ok=True)

        output_path = (
            f"outputs/predictions/"
            f"{model_key}_predictions.csv"
        )

        results_df.to_csv(output_path, index=False)

        # Upload prediction CSV to MLflow
        mlflow.log_artifact(output_path)

        print("\nEvaluation completed!")

        print("\nClassification metrics:")

        for metric_name, metric_value in metrics.items():
            print(f"{metric_name}: {metric_value:.4f}")

        print("\nMLflow Run ID:", run.info.run_id)

        # Save the Hugging Face model as an MLflow model
        print("\nLogging model artifact to MLflow...")

        mlflow.pyfunc.log_model(
            name="sentiment_model",
            python_model=HuggingFaceSentimentModel(model_id),
            signature=MODEL_SIGNATURE,
            pip_requirements=[
                "mlflow==3.18.0",
                "transformers==5.23.0",
                "torch==2.15.0",
                "pandas==3.1.1"
            ]
        )

        print("MLflow model artifact saved successfully!")

        # Release model memory
        del classifier
        del raw_predictions
        gc.collect()

        return {
            "run_id": run.info.run_id,
            "model_key": model_key,
            "model_id": model_id,
            "metrics": metrics,
            "predictions": results_df
        }
    
# --------------------------------------------------
# 9. RUN BOTH MODELS AND COMPARE THEM
# --------------------------------------------------

def main():

    print("\n" + "=" * 60)
    print("CHAMPION-CHALLENGER MODEL EVALUATION")
    print("=" * 60)

    # Load the evaluation dataset
    df = load_evaluation_data()

    # Evaluate Model A first
    result_a = evaluate_model(
        "A",
        MODELS["A"],
        df
    )

    # Evaluate Model B second
    result_b = evaluate_model(
        "B",
        MODELS["B"],
        df
    )

    # Extract Macro-F1 scores
    f1_a = result_a["metrics"]["f1_macro"]
    f1_b = result_b["metrics"]["f1_macro"]

    # Calculate improvement
    delta = f1_b - f1_a

    # Extract average inference latency
    latency_a = result_a["metrics"]["avg_latency_ms"]
    latency_b = result_b["metrics"]["avg_latency_ms"]

    # Apply the pre-defined Champion decision rule
    margin = 0.01

    challenger_wins = (
        delta >= margin
        and latency_b <= 5 * latency_a
    )

    if challenger_wins:
        winner = "B"
        winner_name = MODELS["B"]["model_id"]
    else:
        winner = "A"
        winner_name = MODELS["A"]["model_id"]

    # Display comparison
    print("\n" + "=" * 60)
    print("CHAMPION-CHALLENGER COMPARISON")
    print("=" * 60)

    print(f"Model A Macro-F1: {f1_a:.4f}")
    print(f"Model B Macro-F1: {f1_b:.4f}")
    print(f"Macro-F1 improvement (B - A): {delta:.4f}")

    print(f"\nModel A latency: {latency_a:.2f} ms/example")
    print(f"Model B latency: {latency_b:.2f} ms/example")

    print("\nSelected Champion:", winner)
    print("Champion model ID:", winner_name)

    # Create summary run in MLflow
    with mlflow.start_run(
        run_name="compare_sentiment"
    ):

        mlflow.log_params({
            "primary_metric": "f1_macro",
            "margin": margin,
            "latency_limit_multiplier": 5,
            "model_a_run_id": result_a["run_id"],
            "model_b_run_id": result_b["run_id"]
        })

        mlflow.log_metric(
            "delta_primary_metric",
            delta
        )

        mlflow.set_tag("winner", winner)
        mlflow.set_tag("winner_model_id", winner_name)
        mlflow.set_tag("student_usn", STUDENT_USN)

        # Save comparison results as JSON
        os.makedirs("outputs", exist_ok=True)

        summary = {
            "student_usn": STUDENT_USN,
            "dataset": "SST-2",
            "evaluation_samples": len(df),
            "primary_metric": "f1_macro",
            "margin": margin,
            "model_a": {
                "model_id": result_a["model_id"],
                "run_id": result_a["run_id"],
                "metrics": result_a["metrics"]
            },
            "model_b": {
                "model_id": result_b["model_id"],
                "run_id": result_b["run_id"],
                "metrics": result_b["metrics"]
            },
            "delta_primary_metric": delta,
            "winner": winner,
            "winner_model_id": winner_name
        }

        with open(
            "outputs/metrics_summary.json",
            "w",
            encoding="utf-8"
        ) as file:
            json.dump(summary, file, indent=4)

        mlflow.log_artifact(
            "outputs/metrics_summary.json"
        )

    print("\nComparison results saved successfully!")
    print("File: outputs/metrics_summary.json")
    print("\nBoth model evaluations are recorded in MLflow.")


# --------------------------------------------------
# 10. PROGRAM ENTRY POINT
# --------------------------------------------------

if __name__ == "__main__":
    main()



