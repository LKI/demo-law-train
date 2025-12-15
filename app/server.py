from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from app.inference import stream_response
from app.comparison import stream_compare
import os

app = FastAPI()


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
