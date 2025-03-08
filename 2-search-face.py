import numpy as np
import faiss
import pickle
from deepface import DeepFace
import cv2

index = faiss.read_index("faiss_index.bin")
with open("identities.pkl", "rb") as f:
    identities = pickle.load(f)

# New image path to search
image_path = r'C:/python/1.jpg'

model_name = "ArcFace"

try:
    embedding = DeepFace.represent(image_path, model_name=model_name, enforce_detection=False)[0]["embedding"]
    embedding = np.array(embedding, dtype="float32").reshape(1, -1)

    k = 5  
    distances, indices = index.search(embedding, k)

    print("\n🎯 Top 5 best matches:")
    for i in range(k):
        print(f"{i+1}. {identities[indices[0][i]]} → Distance: {distances[0][i]}")

except Exception as e:
    print(f"❌ Error: {e}")
