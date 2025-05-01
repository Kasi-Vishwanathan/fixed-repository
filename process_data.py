import gzip
import os
import json
import random
from more_itertools import chunked

DATA_DIR = os.path.join(os.path.dirname(__file__), '../data/codesearch')

def format_str(string):
    for char in ['\r\n', '\r', '\n']:
        string = string.replace(char, ' ')
    return string

def preprocess_test_data(language, test_batch_size=1000):
    path = os.path.join(DATA_DIR, f'{language}_test_0.jsonl.gz')
    print(f"Processing file: {path}")
    
    # Read and process data
    processed_data = []
    with gzip.open(path, 'rt') as f:
        for line in f:
            try:
                obj = json.loads(line)
                # Process code content with format_str
                obj['code'] = format_str(obj['code'])
                processed_data.append(json.dumps(obj))
            except json.JSONDecodeError:
                print(f"Warning: Failed to parse line: {line}")
                continue
    
    # Shuffle data using standard library random
    random.seed(0)
    random.shuffle(processed_data)
    
    # Create batches
    batched_data = chunked(processed_data, test_batch_size)
    
    print(f"Processing {len(processed_data)} items into batches")
    for batch_idx, batch in enumerate(batched_data):
        # Process all batches including the last smaller one
        if not batch:
            continue
        print(f"Processing batch {batch_idx} with {len(batch)} items")
        # Add actual batch processing logic here
    
    return processed_data