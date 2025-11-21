# CURATED ML Worker (API Mode)

This is the server-side worker for image captioning + CLIP embeddings.
It runs on Render Free Tier using a FastAPI backend and external API calls
(HuggingFace Inference API) for embeddings and captions.

## Endpoints

### POST /process
Payload:
```json
{
  "artwork_id": "...",
  "storage_url": "https://..."
}
