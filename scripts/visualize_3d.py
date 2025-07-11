import chromadb
import numpy as np
import pandas as pd
import umap
import plotly.express as px

client = chromadb.PersistentClient(path="db")
collection = client.get_collection("memory")       # ← pick your collection name

records = collection.get(
    include=["embeddings", "documents", "metadatas"]       
)
embeds = np.array(records["embeddings"])                   # shape: (n_items, dim)

umap3d = umap.UMAP(n_components=3, metric="cosine", random_state=42)
coords = umap3d.fit_transform(embeds)                      # shape: (n_items, 3)

df = pd.DataFrame(coords, columns=["x", "y", "z"]) # type: ignore
df["label"] = [m["data"] if "data" in m else "No text" for m in records["metadatas"]] # type: ignore
df["short_label"] = [" ".join(m["data"].split()[:1]) for m in records["metadatas"]]   # type: ignore          
df["ids"] = records["ids"]                 

fig = px.scatter_3d(
    df,
    x="x", y="y", z="z",
    hover_name="label",
    text="short_label",         
    # color="ids",            
    opacity=0.4
)
fig.update_layout(
    title="ChromaDB Embeddings (UMAP-3D)",
    margin=dict(l=0, r=0, b=0, t=40)
)
fig.show()
