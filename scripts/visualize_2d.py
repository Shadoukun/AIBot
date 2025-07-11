import chromadb
import umap
import matplotlib.pyplot as plt
import numpy as np
from pprint import pprint

client = chromadb.PersistentClient(path="db")
collection = client.get_collection(name="memory")

results = collection.get(include=["embeddings", "metadatas", "documents"])
pprint(results)
embeddings = results["embeddings"]
documents = results["metadatas"]
if documents is None:
    documents = []

for doc in documents:
    if "data" not in doc:
        doc["data"] = "No text" # type: ignore
    else:
        doc["data"] = " ".join(str(doc["data"]).split()) # type: ignore

labels = [" ".join(doc.get("data", "No text").split()[:2]) for doc in documents] # type: ignore
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine')
embedding_2d = reducer.fit_transform(embeddings)
embedding_2d = np.array(embedding_2d)

plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], alpha=0.7)  
plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], alpha=0.7)

for i, label in enumerate(labels):  # only show a few to avoid clutter
    plt.text(embedding_2d[i, 0], embedding_2d[i, 1], label, fontsize=8)

plt.title("UMAP projection of ChromaDB embeddings")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.grid(True)
plt.tight_layout()
plt.show()
