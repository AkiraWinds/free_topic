'''
load dataset from huggingface
how to run: cd graphrag_db && python3 data_collection.py

'''

import os
import pandas as pd
from datasets import load_dataset

'''
print("start loading dataset...")
print("Loading from HuggingFace: isaacus/open-australian-legal-corpus")

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("isaacus/open-australian-legal-corpus")

print(f"✓ Dataset loaded successfully!")
print(f"Available splits: {list(ds.keys())}")

# Get the first split (corpus, train, etc.)
split_name = list(ds.keys())[0]
dataset = ds[split_name]
print(f"Using split: '{split_name}', Total rows: {len(dataset)}")

#store dataset into data/raw as json
os.makedirs("../data/raw", exist_ok=True)
output_path = "../data/raw/1raw_corpus.jsonl"

print(f"Saving to {output_path}...")
dataset.to_json(output_path, force_ascii=False)

print(f"✓ Data saved successfully!")
'''

# Filter for primary_legislation only
print("\nFiltering for type='primary_legislation'...")
# read 1raw_corpus.jsonl
dataset = pd.read_json("../data/raw/1raw_corpus.jsonl", lines=True)
filtered_dataset = dataset[dataset['type'] == 'primary_legislation']
print(f"Filtered rows: {len(filtered_dataset)}")

# Sort by date (newest to oldest)
print("Sorting by date (newest to oldest)...")
filtered_dataset = filtered_dataset.sort_values('date', ascending=False)

# Save filtered and sorted data
filtered_output_path = "../data/raw/2primary_legislation_sorted.jsonl"
print(f"Saving filtered data to {filtered_output_path}...")
filtered_dataset.to_json(filtered_output_path, orient='records', lines=True, force_ascii=False)

print(f"✓ Filtered data saved successfully!")
print("\nFiltered dataset preview:")
print(filtered_dataset.head(3))  # Display first 3 rows
print(f"\nColumn names: {filtered_dataset.columns.tolist()}")
print(f"Filtered dataset shape: {filtered_dataset.shape}")
