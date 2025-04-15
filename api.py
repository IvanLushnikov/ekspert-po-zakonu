import os
import json
import faiss
import openai
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins="*", methods=["GET", "POST", "OPTIONS"])

@app.after_request
def after_request(response):
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization")
    response.headers.add("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
    return response

# === OpenAI client ===
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# === Загрузка FAISS и метаданных ===
INDEX_PATH = "laws.index"
META_PATH = "laws_meta.json"

index = faiss.read_index(INDEX_PATH)
with open(META_PATH, encoding="utf-8") as f:
    metadata = json.load(f)

# === Функция получения эмбеддинга ===
def get_embedding(text):
    text = text.replace("\n", " ")
    result = client.embeddings.create(
        input=[text],
        model="text-embedding-ada-002"
    )
    return np.array(result.data[0].embedding, dtype="float32")

# === Роут /ask с поиском по RAG ===
@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    user_question = data.get("question")

    if not user_question:
        return jsonify({"error": "No question provided"}), 400

    # --- Поиск по FAISS ---
    query_vec = get_embedding(user_question).reshape(1, -1)
    D, I = index.search(query_vec, k=3)
    context_parts = [metadata[i]["text"] for i in I[0]]
    context = "\n\n".join(context_parts)

    # --- Запрос к GPT ---
    messages = [
        {"role": "system", "content": "Ты — эксперт по государственным закупкам РФ. Отвечай строго по закону, но простыми словами. Ссылайся на статьи, если они есть."},
        {"role": "user", "content": f"Контекст:\n{context}\n\nВопрос: {user_question}"}
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=messages,
            temperature=0.3
        )
        answer = response.choices[0].message.content
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
