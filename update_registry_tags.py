
from mlflow import MlflowClient

TRACKING_URI = "http://127.0.0.1:5000"
REGISTERED_MODEL_NAME = "sentiment-model"

client = MlflowClient(tracking_uri=TRACKING_URI)

# Add Hugging Face model IDs to registered versions
client.set_model_version_tag(
    REGISTERED_MODEL_NAME,
    "1",
    "hf_model_id",
    "distilbert-base-uncased-finetuned-sst-2-english"
)

client.set_model_version_tag(
    REGISTERED_MODEL_NAME,
    "2",
    "hf_model_id",
    "textattack/bert-base-uncased-SST-2"
)

print("Hugging Face model IDs saved successfully!")

# Verify the tags
for version in ["1", "2"]:
    model = client.get_model_version(
        REGISTERED_MODEL_NAME,
        version
    )
    print(f"\nVersion: {version}")
    print(f"Role: {model.tags['role']}")
    print(f"HF Model ID: {model.tags['hf_model_id']}")
