import openai
import faiss
import numpy as np

def get_embedding(text, openai_api_key):
    openai.api_key = openai_api_key
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return np.array(response['data'][0]['embedding'], dtype=np.float32)

def build_faiss_index(drug_data, openai_api_key):
    embeddings = []
    meta = []
    for item in drug_data:
        text = f"{item['itemName']} {item.get('efcyQesitm', '')} {item.get('useMethodQesitm', '')} {item.get('atpnQesitm', '')} {item.get('seQesitm', '')}"
        emb = get_embedding(text, openai_api_key)
        embeddings.append(emb)
        meta.append(item)
    embeddings = np.vstack(embeddings)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, meta