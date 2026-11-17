import pandas as pd
from transformers import pipeline
import gc

# Load 10 examples for initial testing
df = pd.read_csv("data/eval_inputs.csv").head(10)
texts = df["text"].tolist()

models = {
    "Model A": "distilbert-base-uncased-finetuned-sst-2-english",
    "Model B": "textattack/bert-base-uncased-SST-2"
}

for model_name, model_id in models.items():

    print("\n" + "=" * 60)
    print("Testing:", model_name)
    print("Model ID:", model_id)
    print("=" * 60)

    # Load model on CPU
    classifier = pipeline(
        "sentiment-analysis",
        model=model_id,
        tokenizer=model_id,
        device=-1
    )

    # Generate predictions for 10 sentences
    predictions = classifier(
        texts,
        batch_size=2,
        truncation=True,
        max_length=256
    )

    # Display predictions
    for i, prediction in enumerate(predictions):
        print(f"\nExample {i + 1}")
        print("Text:", texts[i][:100])
        print("Prediction:", prediction["label"])
        print("Confidence:", round(prediction["score"], 4))

    # Release memory before loading the next model
    del classifier
    del predictions
    gc.collect()

print("\nBoth model tests completed!")
