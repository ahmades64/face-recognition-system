Face Recognition and Search System
This project is a Face Recognition and Search System that uses advanced deep learning models to store face embeddings in a database and perform fast searches to find the best matches for a given face image. It leverages the DeepFace library for face recognition and FAISS for efficient similarity search.

Features
Face Embedding Storage: Stores face embeddings in a FAISS index for fast retrieval.
Face Search: Searches for the top matches of a given face image in the database.
High Accuracy: Utilizes the ArcFace model for high-precision face recognition.
Scalable: Designed to handle large datasets efficiently.

How It Works
1. Storing Faces in the Database
The script store_faces.py processes images from a specified directory, extracts face embeddings using the ArcFace model, and stores them in a FAISS index. It also saves the corresponding identities (image names) in a pickle file.
2. Searching for Faces
The script search_face.py takes a new face image, extracts its embedding, and searches the FAISS index to find the top 5 most similar faces from the database. It outputs the names of the matched images along with their similarity distances.

Installation
Clone the repository:
git clone https://github.com/your-username/face-recognition-system.git

Install the required dependencies:
pip install -r requirements.txt

Usage
Storing Faces
Place your face images in the C:/python/picture directory.

Run the store_faces.py script:
python store_faces.py

Searching for Faces
Place the image you want to search in the C:/python/ directory (e.g., 1.jpg).

Run the search_face.py script:
python search_face.py
Requirements
Python 3.x

Libraries: deepface, faiss, numpy, pickle, opencv-python
License
This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgments
Thanks to the developers of DeepFace and FAISS for their amazing libraries.

Inspired by various research papers and open-source projects on face recognition.
