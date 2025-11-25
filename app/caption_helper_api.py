# app/caption_helper_api.py
import requests
import base64
import os

class CaptionHelperAPI:
    def __init__(self, hf_token: str):
        if not hf_token:
            raise ValueError("HF_TOKEN is required")
        
        self.hf_token = hf_token
        self.router_url = "https://router.huggingface.co/inference"
        self.model = "Salesforce/blip-image-captioning-base"

    def describe_image(self, image_path: str):
        payload = self._build_payload(image_path)

        headers = {
            "Authorization": f"Bearer {self.hf_token}",
            "Content-Type": "application/json"
        }

        resp = requests.post(
            self.router_url,
            json={
                "model": self.model,
                "inputs": payload
            },
            headers=headers,
            timeout=60
        )

        if resp.status_code != 200:
            return f"Error {resp.status_code}: {resp.text}"

        data = resp.json()

        # Standard HF router response:
        try:
            return data[0]["generated_text"]
        except:
            return str(data)

    def extract_tags(self, image_path: str):
        caption = self.describe_image(image_path)
        tags = [t.lower().strip() for t in caption.replace(".", "").split()]
        return tags

    def embed_image(self, image_path: str):
        return None

    def _build_payload(self, image_path: str):
        # If URL
        if image_path.startswith("http://") or image_path.startswith("https://"):
            return {"image": image_path}

        # Else local file
        with open(image_path, "rb") as f:
            img_bytes = f.read()

        img_b64 = base64.b64encode(img_bytes).decode("utf-8")
        return {"image": img_b64}
