# examples/clear_ltm.py

from qdrant_client import QdrantClient
from app.config import settings

client = QdrantClient(host=settings.LTM_QDRANT_HOST, port=settings.LTM_QDRANT_PORT)
client.delete_collection(collection_name=settings.LTM_COLLECTION_NAME)
print(f"✅ Deleted collection: {settings.LTM_COLLECTION_NAME}")
