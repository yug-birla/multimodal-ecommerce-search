import time
from collections import deque
from fastapi import FastAPI, HTTPException, File, UploadFile, Form, Header
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os

from app.services.search_service import SearchService

# 1. Initialize the App & Memory Queues
app = FastAPI(title="Multimodal E-Commerce Search", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

search_service = SearchService()

# In-memory queues for telemetry (Keep last 50 events to prevent memory leaks)
system_logs = deque(maxlen=50)
latency_history = deque(maxlen=50)

def log_system_event(event_type: str, details: str, latency_sec: float = 0.0):
    """Calculates metrics and pushes them to the rotating log queue."""
    if latency_sec > 0:
        latency_history.append(latency_sec)
    
    timestamp = time.strftime("%H:%M:%S")
    system_logs.appendleft({
        "time": timestamp,
        "type": event_type,
        "details": details,
        "latency_ms": round(latency_sec * 1000, 2)
    })

# 3. Data Validation Schema for Text Search
class SearchRequest(BaseModel):
    query: str
    top_k: int = 6

# 4. The Routes
@app.post("/api/search")
def perform_search(request: SearchRequest):
    start_time = time.time()
    try:
        results = search_service.search(text_query=request.query, top_k=request.top_k)
        elapsed = time.time() - start_time
        log_system_event("TEXT_SEARCH", f"Query: '{request.query}' | Results: {len(results)}", elapsed)
        
        return {
            "status": "success",
            "user_query": request.query,
            "results": results
        }
    except Exception as e:
        log_system_event("ERROR", f"Text Search Failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/search/image")
def perform_image_search(
    file: UploadFile = File(...),
    top_k: int = Form(6)
):
    start_time = time.time()
    try:
        results = search_service.search_image(image_file=file.file, top_k=top_k)
        elapsed = time.time() - start_time
        log_system_event("VISION_SEARCH", f"Image: {file.filename} | Results: {len(results)}", elapsed)
        
        return {
            "status": "success",
            "search_type": "visual",
            "filename": file.filename,
            "results": results
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        log_system_event("ERROR", f"Vision Search Failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@app.post("/api/search/composed")
def perform_composed_search(
    file: UploadFile = File(...),
    query: str = Form(...),
    top_k: int = Form(6)
):
    start_time = time.time()
    try:
        results = search_service.search_composed(image_file=file.file, text_query=query, top_k=top_k)
        elapsed = time.time() - start_time
        log_system_event("HYBRID_SEARCH", f"Image + '{query}' | Results: {len(results)}", elapsed)
        
        return {
            "status": "success",
            "search_type": "hybrid",
            "translated_query": query,
            "results": results
        }
    except Exception as e:
        log_system_event("ERROR", f"Hybrid Search Failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stats")
def get_system_stats(x_admin_token: Optional[str] = Header(None)):
    """Secured Endpoint: Now includes live logs and latency averages."""
    SECRET_ADMIN_TOKEN = os.getenv("DASHBOARD_ADMIN_KEY", "yug_secure_99x")
    
    if not x_admin_token or x_admin_token != SECRET_ADMIN_TOKEN:
        log_system_event("SECURITY", "Blocked unauthorized dashboard access attempt.")
        raise HTTPException(status_code=403, detail="Unauthorized: Access Denied.")
        
    try:
        collection_info = search_service.qdrant.get_collection(collection_name=search_service.collection_name)
        points_count = getattr(collection_info, "points_count", 0)
        qdrant_status = getattr(collection_info, "status", "green")
        
        redis_status = "OFFLINE"
        cache_hits = 0
        cache_misses = 0
        
        if search_service.redis_client is not None:
            redis_status = "CONNECTED"
            try:
                stats = search_service.redis_client.info("stats")
                cache_hits = stats.get("keyspace_hits", 0)
                cache_misses = stats.get("keyspace_misses", 0)
            except Exception as cache_err:
                print(f"⚠️ Telemetry fallback: {cache_err}")

        # Calculate average latency
        avg_latency = (sum(latency_history) / len(latency_history)) if latency_history else 0.0

        return {
            "status": "success",
            "qdrant": {
                "collection": search_service.collection_name,
                "points_count": points_count,
                "status": str(qdrant_status)
            },
            "redis": {
                "status": redis_status,
                "hits": cache_hits,
                "misses": cache_misses
            },
            "performance": {
                "avg_latency_ms": round(avg_latency * 1000, 2),
                "recent_logs": list(system_logs)
            }
        }
    except Exception as e:
        print(f"❌ Telemetry Collection Failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 5. Static File Asset Routing
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_index():
    return FileResponse("static/index.html")