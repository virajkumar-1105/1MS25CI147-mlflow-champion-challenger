from datasets import load_dataset
import pandas as pd

# 1. Project configuration
DATASET_NAME = "nyu-mll/glue"
DATASET_CONFIG = "sst2"
SAMPLE_SIZE = 200
RANDOM_SEED = 42

# 2. Download the validation split
print("Downloading SST-2 dataset...")

dataset = load_dataset(
    DATASET_NAME,
    DATASET_CONFIG,
    split="validation"
)

print("Dataset downloaded successfully!")
print("Available validation examples:", len(dataset))

# 3. Select a reproducible sample
sample = dataset.shuffle(
    seed=RANDOM_SEED
).select(range(SAMPLE_SIZE))

# 4. Create a DataFrame
df = pd.DataFrame({
    "text": sample["sentence"],
    "label": sample["label"]
})

# 5. Save the shared evaluation dataset
df.to_csv("data/eval_inputs.csv", index=False)

# 6. Display the dataset summary
print("\nEvaluation dataset created!")
print("Number of examples:", len(df))

print("\nLabel distribution:")
print(df["label"].value_counts().sort_index())

print("\nFirst five examples:")
print(df.head())

print("\nSaved to data/eval_inputs.csv")
