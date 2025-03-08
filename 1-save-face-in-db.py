import os
import numpy as np
import faiss
import pickle
from deepface import DeepFace

db_path = r'C:/python/picture'

model_name = "ArcFace"

embeddings = []
identities = []

for img_name in os.listdir(db_path):
    img_path = os.path.join(db_path, img_name)
    try:
        embedding = DeepFace.represent(img_path, model_name=model_name, enforce_detection=False)[0]["embedding"]
        embeddings.append(embedding)
        identities.append(img_name)
        print(f"✅ Processed: {img_name}")
    except Exception as e:
        print(f"⚠️ Error processing {img_name}: {e}")

embeddings = np.array(embeddings, dtype="float32")

dimension = embeddings.shape[1]  
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)  

faiss.write_index(index, "faiss_index.bin")

with open("identities.pkl", "wb") as f:
    pickle.dump(identities, f)

print("✅ FAISS database saved successfully!")
