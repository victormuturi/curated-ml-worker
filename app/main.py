# app/main.py
import os
import requests
import tempfile
from fastapi import FastAPI
from pydantic import BaseModel
from supabase import create_client
from .caption_helper_api import CaptionHelperAPI

# Load environment variables
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
HF_TOKEN = os.environ.get("HF_TOKEN")

# Initialize services
supabase = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL and SUPABASE_KEY else None
caption_helper = CaptionHelperAPI(hf_token=HF_TOKEN)

app = FastAPI()

class Job(BaseModel):
    artwork_id: str
    storage_url: str

@app.post("/process")
async def process(job: Job):
    print(f"Processing job for artwork: {job.artwork_id}")
    print(f"Image URL/Path: {job.storage_url}")

    local_path = None
    temp_file = None

    try:
        # Download if URL
        if job.storage_url.startswith("http"):
            print("Downloading image from URL...")
            resp = requests.get(job.storage_url, timeout=30)
            resp.raise_for_status()

            temp_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            temp_file.write(resp.content)
            temp_file.close()

            local_path = temp_file.name
        else:
            local_path = job.storage_url

        print("Generating caption...")
        caption = caption_helper.describe_image(local_path)

        print("Extracting tags...")
        tags = caption_helper.extract_tags(local_path)

        # Embedding placeholder
        embedding = caption_helper.embed_image(local_path)

        # Update Supabase
        if supabase:
            update_payload = {
                "auto_caption": caption,
                "auto_tags": tags,
                "metadata_done": True
            }
            if embedding:
                update_payload["clip_embedding"] = embedding

            supabase.table("artworks").update(update_payload).eq("id", job.artwork_id).execute()

        return {
            "status": "ok",
            "artwork_id": job.artwork_id,
            "caption": caption,
            "tags": tags
        }

    except Exception as e:
        print(f"Error: {e}")
        return {"status": "error", "msg": str(e)}
    
    finally:
        if temp_file and os.path.exists(temp_file.name):
            os.unlink(temp_file.name)
