import json
import tiktoken # for token counting
import numpy as np
from collections import defaultdict

data_path = "../dataset/prompt-train.jsonl"
counter = 0

try:
    # Load the dataset
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            dataset = json.loads(line)
            counter += 1
except Exception as e:
    print(f"Error occurred while reading dataset: {e}")

# Initial dataset stats
print("Num examples:", len(dataset))
print("First example:")
for message in dataset[0]["messages"]:
    print(message)