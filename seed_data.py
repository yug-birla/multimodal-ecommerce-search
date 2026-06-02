import torch
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from transformers import CLIPProcessor, CLIPModel

print("1. Loading AI Model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_id = "openai/clip-vit-base-patch32"
processor = CLIPProcessor.from_pretrained(clip_id)
model = CLIPModel.from_pretrained(clip_id).to(device)

client = QdrantClient(url="http://localhost:6333")

dummy_items = [
    {"id": 1, "title": "Minimalist Ceramic Coffee Mug", "price": 14.99, "category": "Home & Kitchen", "image_url": "https://images.unsplash.com/photo-1514228742587-6b1558fcca3d?w=500"},
    {"id": 2, "title": "Matte Black Espresso Cup", "price": 12.00, "category": "Home & Kitchen", "image_url": "https://images.unsplash.com/photo-1576697440115-c38a5b28e219?w=500"},
    {"id": 3, "title": "Leather Aviator Jacket", "price": 120.00, "category": "Fashion", "image_url": "https://images.unsplash.com/photo-1551028719-00167b16eac5?w=500"},
    {"id": 4, "title": "Vintage Denim Jacket", "price": 85.00, "category": "Fashion", "image_url": "https://images.unsplash.com/photo-1601333144130-8cbb312386b6?w=500"},
    {"id": 5, "title": "Camping Tent for 4 Persons", "price": 89.99, "category": "Sports & Outdoors", "image_url": "https://images.unsplash.com/photo-1478131143081-80f7f84ca84d?w=500"},
    {"id": 6, "title": "Extreme Cold Sleeping Bag", "price": 150.00, "category": "Sports & Outdoors", "image_url": "https://images.unsplash.com/photo-1536768139911-e290a59011e4?w=500"},
    {"id": 7, "title": "Luxury Mechanical Men's Watch", "price": 299.99, "category": "Accessories", "image_url": "https://images.unsplash.com/photo-1524592094714-0f0654e20314?w=500"},
    {"id": 8, "title": "Professional Running Sneakers", "price": 110.00, "category": "Fashion", "image_url": "https://images.unsplash.com/photo-1542291026-7eec264c27ff?w=500"},
    {"id": 9, "title": "Wireless Noise-Canceling Headphones", "price": 199.99, "category": "Electronics", "image_url": "https://images.unsplash.com/photo-1618366712010-f4ae9c647dcb?w=500"},
    {"id": 10, "title": "Rustic Wooden Dining Table", "price": 450.00, "category": "Home & Kitchen", "image_url": "https://images.unsplash.com/photo-1533090481720-856c6e3c1fdc?w=500"}
]

print("2. Generating Real Semantic Vectors...")
points = []
for item in dummy_items:
    inputs = processor(text=[item["title"]], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        features = model.get_text_features(**inputs)
        
        # --- THE FIX: Extract the raw tensor from the HuggingFace wrapper ---
        if not isinstance(features, torch.Tensor):
            if hasattr(features, "text_embeds"):
                features = features.text_embeds
            elif hasattr(features, "pooler_output"):
                features = features.pooler_output
            else:
                features = features[0]
                
        # Now the math will work perfectly!
        features = features / features.norm(p=2, dim=-1, keepdim=True)
        real_vector = features.cpu().numpy()[0].tolist()

    points.append(PointStruct(id=item["id"], vector=real_vector, payload=item))

client.upsert(collection_name="products", points=points)
print("3. Success! 10 Real products injected into the database.")