from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.inference import stream_response
from app.comparison import stream_compare, load_models
import os

app = FastAPI()

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


@app.on_event("startup")
async def startup_event():
    """
    Load models on startup so the first request isn't slow.
    """
    try:
        load_models()
    except Exception as e:
        print(f"[ERROR] Failed to load models on startup: {e}")


class ChatRequest(BaseModel):
    message: str


@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    return StreamingResponse(stream_response(request.message), media_type="text/plain")


@app.post("/api/compare")
async def compare_endpoint(request: ChatRequest):
    return StreamingResponse(
        stream_compare(request.message), media_type="application/x-ndjson"
    )


# Mount web directory for static files
# Ensure the directory exists to avoid errors on startup if it's not created yet
web_dir = os.path.join(os.getcwd(), "web")
os.makedirs(web_dir, exist_ok=True)

app.mount("/", StaticFiles(directory=web_dir, html=True), name="static")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
