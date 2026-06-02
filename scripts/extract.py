import gzip
import json
import os
import glob
from collections import defaultdict
from tqdm import tqdm

RAW_DIR = "data/raw/"
OUTPUT_FILE = "data/processed/clean_80k_diverse.jsonl"
ITEMS_PER_CATEGORY = 20000
MAX_PER_SUBCATEGORY = 100

def extract_diverse_categories():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    files = glob.glob(os.path.join(RAW_DIR, "*.jsonl.gz"))
    valid_items = []
    
    for file_path in files:
        category_name = os.path.basename(file_path).split('.')[0].replace('meta_', '')
        print(f"\nProcessing category: {category_name}")
        
        category_count = 0
        sub_category_counts = defaultdict(int)
        
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            for line in tqdm(f, desc=f"Scanning {category_name}"):
                try:
                    item = json.loads(line)
                    
                    # 1. Base Validations
                    has_title = item.get('title')
                    has_price = item.get('price') is not None
                    has_images = item.get('images') and isinstance(item['images'], list) and len(item['images']) > 0
                    
                    if has_title and has_price and has_images:
                        image_data = item['images'][0]
                        image_url = image_data.get('large') or image_data.get('hi_res') or image_data.get('thumb')
                        if not image_url: continue
                            
                        # 2. AGGRESSIVE CATEGORY EXTRACTION
                        cats = item.get("category") or item.get("categories")
                        
                        if cats:
                            # Handle lists, nested lists, and raw strings
                            if isinstance(cats, list):
                                if len(cats) > 0 and isinstance(cats[0], list):
                                    sub_category = cats[0][-1]
                                elif len(cats) > 0:
                                    sub_category = cats[-1]
                                else:
                                    sub_category = "Unknown"
                            elif isinstance(cats, str):
                                sub_category = cats
                            else:
                                sub_category = str(cats)
                        else:
                            # DATA ENGINEERING HACK: Generate synthetic category from the Title
                            title_words = item.get("title", "").split()
                            sub_category = " ".join(title_words[:2]) if len(title_words) >= 2 else "Unknown"
                            
                        # THE BOUNCER: Limit to 100 items per specific type
                        if sub_category_counts[sub_category] >= MAX_PER_SUBCATEGORY:
                            continue
                            
                        desc_list = item.get('description', [])
                        description = desc_list[0] if desc_list else ""
                        
                        clean_item = {
                            "id": item.get("parent_asin", ""),
                            "title": item.get("title", ""),
                            "category": category_name, 
                            "sub_category": sub_category, 
                            "image_url": image_url,
                            "price": item.get("price"),
                            "description": description
                        }
                        
                        valid_items.append(clean_item)
                        sub_category_counts[sub_category] += 1
                        category_count += 1
                        
                        if category_count >= ITEMS_PER_CATEGORY:
                            break
                except json.JSONDecodeError:
                    continue
        
        print(f"Extracted {category_count} highly diverse items from {len(sub_category_counts)} unique sub-categories.")

    print(f"\nTotal diverse items extracted: {len(valid_items)}")
    print(f"Saving to {OUTPUT_FILE}...")
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as out_f:
        for item in valid_items:
            out_f.write(json.dumps(item) + '\n')
            
    print("Diversity extraction complete! Your balanced dataset is ready.")

if __name__ == "__main__":
    extract_diverse_categories()