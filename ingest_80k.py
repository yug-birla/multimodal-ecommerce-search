import os
import json
import torch
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from transformers import CLIPProcessor, CLIPModel

print("1. Loading AI Model into Memory...")
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_id = "openai/clip-vit-base-patch32"
processor = CLIPProcessor.from_pretrained(clip_id)
model = CLIPModel.from_pretrained(clip_id).to(device)

print("2. Connecting to Qdrant Cloud...")
client = QdrantClient(
    # MAKE SURE YOUR URL LOOKS EXACTLY LIKE THIS (with https:// and :6333)
    url="", 
    port=6333, # Explicitly tell Qdrant to use the REST API port
    api_key="",
    timeout=60.0
)
print("2.5 Checking Database Collection...")
# Tell Qdrant to build the collection specifically for 512-dimensional CLIP vectors
if not client.collection_exists(collection_name="products"):
    print("   -> Collection 'products' not found. Building it now...")
    client.create_collection(
        collection_name="products",
        vectors_config=VectorParams(
            size=512, 
            distance=Distance.COSINE
        )
    )
    print("   -> Collection built successfully!")
else:
    print("   -> Collection 'products' already exists.")

print("3. Reading the JSONL Dataset...")
products = []
file_path = r"E:\multimodal-search\data\processed\clean_80k_multicategory.jsonl"

with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():  # Ignore any accidental blank lines
            products.append(json.loads(line))

total_items = len(products)
BATCH_SIZE = 64  
print(f"4. Beginning Batch Vectorization (Total Items: {total_items})...")

for i in tqdm(range(0, total_items, BATCH_SIZE), desc="Processing Batches"):
    # Slice out a batch of 64 products
    batch = products[i : i + BATCH_SIZE]
    
    # Extract the text fields to feed into the AI model
    # (If your JSON uses "product_name" instead of "title", change it here!)
    titles = [item.get("title", "") for item in batch]
    
    # Generate semantic vectors using CLIP
    inputs = processor(text=titles, return_tensors="pt", padding=True, truncation=True).to(device)
    
    with torch.no_grad():
        features = model.get_text_features(**inputs)
        
        # Unpack the raw tensor from the Hugging Face wrapper
        if not isinstance(features, torch.Tensor):
            if hasattr(features, "text_embeds"):
                features = features.text_embeds
            elif hasattr(features, "pooler_output"):
                features = features.pooler_output
            else:
                features = features[0]
                
        # Normalize vectors for accurate Cosine Similarity mapping
        features = features / features.norm(p=2, dim=-1, keepdim=True)
        batch_vectors = features.cpu().numpy().tolist()

    # Package the items with their metadata payloads
    points = []
    for idx, item in enumerate(batch):
        # (If your JSON uses different keys, update them here!)
        payload = {
            "title": item.get("title", ""),
            "price": item.get("price", 0.0),
            "category": item.get("category", "Uncategorized"),
            "image_url": item.get("image_url", "")
        }
        
        points.append(
            PointStruct(
                id=i + idx + 1, # Creates a sequential unique ID (1 to 80,000)
                vector=batch_vectors[idx],
                payload=payload
            )
        )

    # Stream the vectorized batch up to Qdrant Cloud
    client.upsert(collection_name="products", points=points)

print("\n✅ Success! The entire dataset has been successfully vectorized and hosted online.")