import os
import openai
import pandas as pd
import numpy as np

openai.api_key = os.getenv("OPENAI_API_KEY")

df = pd.read_csv("knowledge_base_updated.csv", sep=';')

def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    response = openai.embeddings.create(input=[text], model=model)
    return response.data[0].embedding

df["embedding"] = df["question"].apply(get_embedding)

df.to_pickle("knowledge_base_with_embeddings.pkl")
print("✅ Готово! Embedding сохранены в knowledge_base_with_embeddings.pkl")
