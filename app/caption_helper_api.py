# app/caption_helper_api.py
import os
import requests
import json

HF_API_TOKEN = os.environ.get("HF_API_TOKEN")
REPLICATE_TOKEN = os.environ.get("REPLICATE_TOKEN")

class CaptionHelperAPI:

    def __init__(self):
        self.hf_token = HF_API_TOKEN
        self.replicate_token = REPLICATE_TOKEN

    # --------------------------
    # IMAGE CAPTIONING
    # --------------------------
    def describe_image(self, local_path: str) -> str:
        # Use HuggingFace Inference API
        if self.hf_token:
            url = "https://api-inference.huggingface.co/models/llava-hf/llava-1.5-7b"  
            headers = {"Authorization": f"Bearer {self.hf_token}"}

            with open(local_path, "rb") as f:
                data = f.read()

            try:
                res = requests.post(url, headers=headers, data=data, timeout=60)
                if res.status_code == 200:
                    j = res.json()
                    # HF sometimes returns a list of dicts
                    if isinstance(j, list) and "generated_text" in j[0]:
                        return j[0]["generated_text"]
                    return str(j)
                else:
                    return f"Inference error {res.status_code}"
            except Exception as e:
                return f"Caption error: {e}"

        # fallback
        return "No caption: HF token missing."

    # --------------------------
    # IMAGE EMBEDDINGS
    # --------------------------
    def embed_image(self, local_path: str):
        if self.hf_token:
            url = "https://api-inference.huggingface.co/pipeline/feature-extraction/openai/clip-vit-base-patch32"
            headers = {"Authorization": f"Bearer {self.hf_token}"}

            with open(local_path, "rb") as f:
                data = f.read()

            try:
                res = requests.post(url, headers=headers, data=data, timeout=60)
                if res.status_code == 200:
                    vect = res.json()
                    # flatten (HF returns 2D sometimes)
                    flat = []
                    for row in vect:
                        if isinstance(row, list):
                            flat.extend(row)
                        else:
                            flat.append(row)
                    return flat
                return None
            except Exception:
                return None

        return None

    # --------------------------
    # AUTO-TAGGING (simple)
    # --------------------------
    def extract_tags(self, caption: str):
        if not caption:
            return []
        words = caption.lower().replace(",", " ").split()
        tags = [w for w in words if len(w) > 3][:10]
        return tags
