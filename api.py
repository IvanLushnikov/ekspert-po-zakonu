import os
import csv
import openai
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
from pathlib import Path
from numpy.linalg import norm

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

openai.api_key = os.getenv("OPENAI_API_KEY")

# === Загрузка базы с embedding ===
try:
    kb_path = Path(__file__).resolve().parent / "knowledge_base_with_embeddings.pkl"
    df_kb = pd.read_pickle(kb_path)
    df_kb["embedding"] = df_kb["embedding"].apply(np.array)
    print("📚 База знаний с embedding загружена.")
except Exception as e:
    print(f"❌ Ошибка при загрузке базы знаний: {e}")
    df_kb = pd.DataFrame(columns=["question", "answer", "embedding"])

# === Функция для получения embedding ===
def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    response = openai.Embedding.create(input=[text], model=model)
    return np.array(response.data[0].embedding)

# === Поиск ближайших вопросов ===
def search_similar_questions(question, top_n=3):
    query_emb = get_embedding(question)
    df_kb["similarity"] = df_kb["embedding"].apply(lambda x: np.dot(x, query_emb) / (norm(x) * norm(query_emb)))
    top = df_kb.sort_values("similarity", ascending=False).head(top_n)
    return top[["question", "answer"]].to_dict(orient="records")

# === Основной маршрут ===
@app.route("/ask", methods=["POST", "OPTIONS"])
def ask():
    if request.method == "OPTIONS":
        # Явный ответ на preflight-запрос
        response = make_response('', 200)
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        return response

    data = request.get_json()
    user_question = data.get("question", "")
    print("📥 Вопрос пользователя:", user_question)

    try:
        context_items = search_similar_questions(user_question)
    except Exception as e:
        print("❌ Ошибка при поиске embedding:", e)
        context_items = []

    context = "\n\n".join([f"Вопрос: {i['question']}\nОтвет: {i['answer']}" for i in context_items])
    print("📚 Контекст найден:", context != "")

    if context.strip():
        prompt = f"Ты эксперт по 44-ФЗ. Используй контекст ниже, чтобы ответить на вопрос:\n\n{context}\n\nВопрос: {user_question}"
    else:
        prompt = f"Ты эксперт по 44-ФЗ. Ответь на вопрос пользователя максимально полно:\n\nВопрос: {user_question}"

    try:
        print("🧪 Запрос в OpenAI...")
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.7,
        )
        answer = response.choices[0].message.content
        response = jsonify({"answer": answer})
        response.headers.add("Access-Control-Allow-Origin", "*")
        return response

    except Exception as e:
        print("❌ Ошибка OpenAI:", e)
        response = jsonify({"error": str(e)})
        response.headers.add("Access-Control-Allow-Origin", "*")
        return response, 500

# === Проверка ключа ===
@app.route("/check-key", methods=["GET"])
def check_key():
    try:
        models = openai.models.list()
        return jsonify({"status": "ok", "models": [m.id for m in models.data]})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 401

# === Запуск локально ===
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(debug=True, host='0.0.0.0', port=port)
