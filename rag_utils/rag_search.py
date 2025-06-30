import faiss
import openai
import json
import numpy as np
import configparser
import pandas as pd
import sys
import os

def get_embedding(text, openai_api_key):
    openai.api_key = openai_api_key
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return np.array(response['data'][0]['embedding'], dtype=np.float32)

def load_index_and_meta(index_path, meta_path):
    index = faiss.read_index(index_path)
    with open(meta_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    return index, meta

def search_similar_diseases(symptom_list, index_path, meta_path, top_k=3):
    config = configparser.ConfigParser()
    config.read('keys.config')
    openai_api_key = config['API_KEYS']['chatgpt_api_key']
    text = ', '.join(symptom_list)
    query_emb = get_embedding(text, openai_api_key).reshape(1, -1)
    index, meta = load_index_and_meta(index_path, meta_path)
    D, I = index.search(query_emb, top_k)
    results = [meta[i] for i in I[0]]
    return results

def evaluate_rag(test_csv, index_path, meta_path, top_k=3):
    config = configparser.ConfigParser()
    config.read('keys.config')
    openai_api_key = config['API_KEYS']['chatgpt_api_key']
    df = pd.read_csv(test_csv)
    symptom_cols = [col for col in df.columns if col.lower() != 'disease']
    correct = 0
    total = 0
    for _, row in df.iterrows():
        symptoms_present = [col for col in symptom_cols if row[col] == 1]
        true_disease = row['Disease']
        results = search_similar_diseases(symptoms_present, index_path, meta_path, top_k=top_k)
        if results and results[0]['disease'] == true_disease:
            correct += 1
        total += 1
    accuracy = correct / total if total > 0 else 0
    print(f"Top-1 Accuracy (Top-{top_k} retrieved): {accuracy:.4f} ({correct}/{total})")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python rag_search.py <test_csv> <index_path> <meta_path>")
        sys.exit(1)
    test_csv = sys.argv[1]
    index_path = sys.argv[2]
    meta_path = sys.argv[3]
    evaluate_rag(test_csv, index_path, meta_path, top_k=3) 