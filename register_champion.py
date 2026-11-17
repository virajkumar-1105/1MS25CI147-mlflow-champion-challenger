import mlflow
from mlflow import MlflowClient

# --------------------------------------------------
# 1. PROJECT CONFIGURATION
# --------------------------------------------------

TRACKING_URI = "http://127.0.0.1:5000"

EXPERIMENT_NAME = "hf-sentiment-champion-challenger"

REGISTERED_MODEL_NAME = "sentiment-model"

STUDENT_USN = "1MS25CI147"


# --------------------------------------------------
# 2. CONNECT TO MLFLOW
# --------------------------------------------------

mlflow.set_tracking_uri(TRACKING_URI)

client = MlflowClient(
    tracking_uri=TRACKING_URI
)

experiment = mlflow.get_experiment_by_name(
    EXPERIMENT_NAME
)

if experiment is None:
    raise ValueError("MLflow experiment not found!")

print("Connected to MLflow successfully!")
print("Experiment:", EXPERIMENT_NAME)
print("Student USN:", STUDENT_USN)
print("Registry name:", REGISTERED_MODEL_NAME)

# --------------------------------------------------
# 3. FIND THE LATEST SUCCESSFUL MODEL RUN
# --------------------------------------------------

def find_latest_run(run_name):

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=(
            f"tags.mlflow.runName = '{run_name}'"
        ),
        order_by=["attributes.start_time DESC"],
        max_results=10
    )

    # Select the latest completed run
    for run in runs:

        if run.info.status == "FINISHED":
            return run

    raise ValueError(
        f"No completed run found for {run_name}"
    )


print("\nSearching for the latest model runs...")

run_a = find_latest_run(
    "A_sentiment_distilbert-sst2"
)

run_b = find_latest_run(
    "B_sentiment_bert-sst2"
)

print("\nModel A Run ID:", run_a.info.run_id)
print("Model B Run ID:", run_b.info.run_id)

# --------------------------------------------------
# 4. FIND LOGGED MODEL IDs
# --------------------------------------------------

def find_logged_model(run_id):

    logged_models = mlflow.search_logged_models(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"source_run_id = '{run_id}'",
        output_format="list"
    )

    # Find the model named sentiment_model
    for model in logged_models:

        if model.name == "sentiment_model":
            return model

    raise ValueError(
        f"No sentiment_model found for run {run_id}"
    )


print("\nSearching for logged model artifacts...")

model_a = find_logged_model(run_a.info.run_id)

model_b = find_logged_model(run_b.info.run_id)

print("\nModel A logged-model ID:", model_a.model_id)
print("Model B logged-model ID:", model_b.model_id)

# --------------------------------------------------
# 5. REGISTER BOTH MODELS
# --------------------------------------------------

print("\nRegistering Model A...")

version_a = mlflow.register_model(
    model_uri=f"models:/{model_a.model_id}",
    name=REGISTERED_MODEL_NAME,
    await_registration_for=300
)

print("Model A registered!")
print("Model A version:", version_a.version)


print("\nRegistering Model B...")

version_b = mlflow.register_model(
    model_uri=f"models:/{model_b.model_id}",
    name=REGISTERED_MODEL_NAME,
    await_registration_for=300
)

print("Model B registered!")
print("Model B version:", version_b.version)

# --------------------------------------------------
# 6. ASSIGN CHAMPION AND CHALLENGER ALIASES
# --------------------------------------------------

print("\nAssigning model aliases...")

# Model B is the Champion
client.set_registered_model_alias(
    REGISTERED_MODEL_NAME,
    "champion",
    version_b.version
)

# Model A is the Challenger
client.set_registered_model_alias(
    REGISTERED_MODEL_NAME,
    "challenger",
    version_a.version
)


# Add student and role information to model versions
client.set_model_version_tag(
    REGISTERED_MODEL_NAME,
    version_a.version,
    "student_usn",
    STUDENT_USN
)

client.set_model_version_tag(
    REGISTERED_MODEL_NAME,
    version_a.version,
    "role",
    "challenger"
)

client.set_model_version_tag(
    REGISTERED_MODEL_NAME,
    version_b.version,
    "student_usn",
    STUDENT_USN
)

client.set_model_version_tag(
    REGISTERED_MODEL_NAME,
    version_b.version,
    "role",
    "champion"
)


print("\nAliases assigned successfully!")

print("Champion: Model B, Version", version_b.version)
print("Challenger: Model A, Version", version_a.version)

# --------------------------------------------------
# 7. VERIFY MODEL REGISTRY
# --------------------------------------------------

champion = client.get_model_version_by_alias(
    REGISTERED_MODEL_NAME,
    "champion"
)

challenger = client.get_model_version_by_alias(
    REGISTERED_MODEL_NAME,
    "challenger"
)

print("\n" + "=" * 55)
print("MODEL REGISTRY VERIFICATION")
print("=" * 55)

print("\nRegistered Model:", REGISTERED_MODEL_NAME)

print("\nCHAMPION")
print("Version:", champion.version)
print("Model URI:", f"models:/{REGISTERED_MODEL_NAME}@champion")

print("\nCHALLENGER")
print("Version:", challenger.version)
print("Model URI:", f"models:/{REGISTERED_MODEL_NAME}@challenger")

print("\nMODEL REGISTRATION COMPLETED SUCCESSFULLY!")

