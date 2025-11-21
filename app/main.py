# app/main.py
import os
import tempfile
import requests
from fastapi import FastAPI
from pydantic import BaseModel
from supabase import create_client
from .caption_helper_api import CaptionHelperAPI

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
USE_LOCAL_MODEL = os.environ.get("USE_LOCAL_MODEL", "false").lower() == "true"

# Initialize supabase client
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Captions & embeddings via external APIs (Hugging Face / Replicate)
caption_helper = CaptionHelperAPI()

app = FastAPI()

class Job(BaseModel):
    artwork_id: str
    storage_url: str


@app.post("/process")
async def process(job: Job):

    # ---- 1. DOWNLOAD IMAGE FROM SUPABASE STORAGE ----
    try:
        img_response = requests.get(job.storage_url, timeout=30)
        img_response.raise_for_status()
    except Exception as e:
        return {"status": "error", "msg": f"Failed to download image: {e}"}

    # save temp file
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        f.write(img_response.content)
        local_path = f.name

    # ---- 2. GET IMAGE EMBEDDING ----
    embedding = caption_helper.embed_image(local_path)

    # ---- 3. GET IMAGE CAPTION ----
    caption = caption_helper.describe_image(local_path)

    # ---- 4. AUTO TAGGING ----
    tags = caption_helper.extract_tags(caption)

    # ---- 5. UPDATE SUPABASE ROW ----
    update_payload = {
        "auto_caption": caption,
        "auto_tags": tags,
        "metadata_done": True
    }

    if embedding is not None:
        update_payload["clip_embedding"] = embedding

    try:
        supabase.table("artworks").update(update_payload).eq("id", job.artwork_id).execute()
    except Exception as e:
        return {"status": "error", "msg": f"Supabase update failed: {e}"}

    return {
        "status": "ok",
        "artwork_id": job.artwork_id,
        "caption": caption,
        "tags": tags
    }
