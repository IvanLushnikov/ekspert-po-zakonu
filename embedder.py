import os
import openai
import pandas as pd
import numpy as np

openai.api_key = os.getenv("OPENAI_API_KEY")

# Загрузка CSV
df = pd.read_csv("knowledge_base.csv", sep=';')

# Очистка текста и получение эмбеддингов
def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ").strip()
    if not text:
        return np.zeros(1536)  # для пустых строк
    response = openai.embeddings.create(input=[text], model=model)
    return response.data[0].embedding

# Генерация эмбеддингов по вопросам
df["embedding"] = df["question"].apply(get_embedding)

# Сохранение
df.to_pickle("knowledge_base_with_embeddings.pkl")
print("✅ Готово! Embedding сохранены в knowledge_base_with_embeddings.pkl")
