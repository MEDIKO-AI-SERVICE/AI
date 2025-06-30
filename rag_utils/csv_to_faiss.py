import pandas as pd
import numpy as np
import faiss
import openai
import json
import configparser
import sys
import os

def get_embedding(text, openai_api_key):
    openai.api_key = openai_api_key
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return np.array(response['data'][0]['embedding'], dtype=np.float32)

def main(csv_path):
    # Load API key
    config = configparser.ConfigParser()
    config.read('keys.config')
    openai_api_key = config['API_KEYS']['chatgpt_api_key']

    # Load CSV
    df = pd.read_csv(csv_path)
    symptom_cols = [col for col in df.columns if col.lower() != 'disease']
    meta = []
    embeddings = []

    for _, row in df.iterrows():
        symptoms_present = [col for col in symptom_cols if row[col] == 1]
        text = ', '.join(symptoms_present)
        emb = get_embedding(text, openai_api_key)
        embeddings.append(emb)
        meta.append({
            'disease': row['Disease'],
            'symptoms': symptoms_present
        })

    embeddings = np.vstack(embeddings)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    # Save index and meta
    base = os.path.splitext(os.path.basename(csv_path))[0]
    index_path = f'rag_utils/{base}.index'
    meta_path = f'rag_utils/{base}_meta.json'
    faiss.write_index(index, index_path)
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"Saved FAISS index to {index_path} and meta to {meta_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python csv_to_faiss.py <csv_path>")
        sys.exit(1)
    main(sys.argv[1]) 