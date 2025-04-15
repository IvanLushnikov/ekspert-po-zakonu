import os
import openai
import pandas as pd
import numpy as np

openai.api_key = os.getenv("OPENAI_API_KEY")

df = pd.read_csv("knowledge_base.csv", sep=';')

def get_embedding(text, model="text-embedding-3-small"):
    if not isinstance(text, str) or not text.strip():
        return np.zeros(1536) 
    text = text.replace("\n", " ").strip()
    response = openai.Embedding.create(input=[text], model=model)
    return response.data[0].embedding


df["embedding"] = df["question"].apply(get_embedding)


df.to_pickle("knowledge_base_with_embeddings.pkl")
print("Готово")
