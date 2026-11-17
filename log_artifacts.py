import os
from mlflow import MlflowClient

TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "hf-sentiment-champion-challenger"

client = MlflowClient(tracking_uri=TRACKING_URI)

# Find the experiment
experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

if experiment is None:
    raise ValueError("MLflow experiment not found!")

# Find the latest comparison run
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="tags.mlflow.runName = 'compare_sentiment'",
    order_by=["attributes.start_time DESC"],
    max_results=10
)

if not runs:
    raise ValueError("Comparison run not found!")

comparison_run = runs[0]
run_id = comparison_run.info.run_id

print("Comparison run found:", run_id)

# Files to upload
files = [
    ("data/eval_inputs.csv", "data"),
    ("data/new_data.csv", "data"),
    ("outputs/metrics_summary.json", "evaluation"),
    ("outputs/disagreement_examples.csv", "evaluation"),
    (
        "outputs/predictions/new_data_predictions.csv",
        "predictions"
    ),
]

# Upload individual files
for file_path, artifact_folder in files:
    if os.path.exists(file_path):
        client.log_artifact(
            run_id,
            file_path,
            artifact_path=artifact_folder
        )
        print("Uploaded:", file_path)
    else:
        print("WARNING: File not found:", file_path)

# Upload all generated plots
plot_folder = "outputs/plots"

if os.path.isdir(plot_folder):
    client.log_artifacts(
        run_id,
        plot_folder,
        artifact_path="plots"
    )
    print("Uploaded all plots successfully!")
else:
    print("WARNING: Plot folder not found!")

print("\n================================")
print("ARTIFACT UPLOAD COMPLETED!")
print("Comparison Run ID:", run_id)
print("================================")
