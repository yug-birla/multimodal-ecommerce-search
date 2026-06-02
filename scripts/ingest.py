import json
import torch
import requests
from PIL import Image
from io import BytesIO
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

# --- Configuration ---
DATA_FILE = "data/processed/clean_80k_diverse.jsonl" # Pointing to the new 80k dataset
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "products"
BATCH_SIZE = 32 # Maximum batch size for 16GB RAM

# --- Device & Model Setup ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Connecting to Qdrant...")
client = QdrantClient(url=QDRANT_URL)

# Create the database collection if it doesn't exist
if not client.collection_exists(COLLECTION_NAME):
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=512, distance=Distance.COSINE),
    )
    print(f"Created new collection: {COLLECTION_NAME}")

print(f"Loading CLIP Model into Memory ({device.upper()} Mode)...")
model_id = "openai/clip-vit-base-patch32"
processor = CLIPProcessor.from_pretrained(model_id)
model = CLIPModel.from_pretrained(model_id).to(device)
model.eval() # Set model to strict evaluation mode

def download_image(url):
    """Safely download an image. Returns None if the link is dead."""
    try:
        response = requests.get(url, timeout=3)
        if response.status_code == 200:
            return Image.open(BytesIO(response.content)).convert("RGB")
    except Exception:
        pass
    return None

def process_batch(batch_items):
    """Embeds the images and uploads them + text payload to Qdrant."""
    valid_images = []
    valid_items = []
    
    # 1. Download images for this batch
    for item in batch_items:
        img = download_image(item['image_url'])
        if img:
            valid_images.append(img)
            valid_items.append(item)
            
    if not valid_images:
        return # Skip if all URLs in this batch were dead
        
    # 2. INFERENCE: Pass images through CLIP to get 512-dim vectors
    with torch.no_grad():
        inputs = processor(images=valid_images, return_tensors="pt").to(device)
        
        # Explicitly call the vision-only function
        image_features = model.get_image_features(pixel_values=inputs['pixel_values'])
        
        # Bulletproof Check: Extract tensor if Hugging Face returns an object
        if not isinstance(image_features, torch.Tensor):
            if hasattr(image_features, "pooler_output"):
                image_features = image_features.pooler_output
            elif hasattr(image_features, "image_embeds"):
                image_features = image_features.image_embeds
            else:
                image_features = image_features[0]

        # Normalize the vectors for Cosine Similarity
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        
        # Move back to CPU and convert to numpy for Qdrant
        vectors = image_features.cpu().numpy()

    # 3. Prepare data for Qdrant
    points = []
    for i, item in enumerate(valid_items):
        point_id = abs(hash(item['id'])) % (10 ** 15) 
        
        points.append(
            PointStruct(
                id=point_id,
                vector=vectors[i].tolist(),
                payload={
                    "asin": item['id'],
                    "title": item['title'],
                    "category": item['category'],
                    "sub_category": item.get('sub_category', 'Unknown'), # New metadata included
                    "price": item['price'],
                    "image_url": item['image_url'],
                    "description": item['description']
                }
            )
        )
        
    # 4. Upload to Vector Database
    client.upsert(collection_name=COLLECTION_NAME, points=points)

def run_ingestion():
    print("Starting Data Ingestion Pipeline...")
    
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    batch = []
    for line in tqdm(lines, desc="Processing Batches"):
        item = json.loads(line)
        batch.append(item)
        
        if len(batch) >= BATCH_SIZE:
            process_batch(batch)
            batch = []
            
    if batch:
        process_batch(batch)
        
    print("Ingestion Complete! Database is populated.")

if __name__ == "__main__":
    run_ingestion()