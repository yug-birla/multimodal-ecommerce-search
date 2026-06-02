import gzip
import json

# We just want to peek at one file
INPUT_FILE = "data/raw/meta_Home_and_Kitchen.jsonl.gz"

print("Extracting the first raw item...\n")

with gzip.open(INPUT_FILE, 'rt', encoding='utf-8') as f:
    # Read just the very first line
    first_line = f.readline()
    
    # Parse it and print it out nicely formatted
    item = json.loads(first_line)
    print(json.dumps(item, indent=4))