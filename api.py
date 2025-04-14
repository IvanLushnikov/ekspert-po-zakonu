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
CORS(app)

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
    return top["question"].tolist(), top["answer"].tolist(), top["similarity"].tolist()

# === Основной маршрут ===
@app.route("/ask", methods=["POST", "OPTIONS"])
def ask():
    if request.method == "OPTIONS":
        response = make_response()
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return response, 200

    try:
        data = request.get_json()
        user_question = data.get("question", "")
        print("📥 Вопрос пользователя:", user_question)

        try:
            questions, answers, similarities = search_similar_questions(user_question)
        except Exception as e:
            print("❌ Ошибка при поиске embedding:", e)
            questions, answers, similarities = [], [], []

        # Проверка на релевантность первого найденного ответа
        if similarities and similarities[0] > 0.90:
            best_answer = answers[0]
            answer = f"💡 Ответ из базы знаний:\n{best_answer}"
        else:
            # Контекст для запроса в GPT
            context = "\n\n".join([f"Вопрос: {q}\nОтвет: {a}" for q, a in zip(questions, answers)])
            prompt = f"Ты — эксперт по государственным закупкам в РФ.\nИспользуй только информацию из контекста ниже, чтобы ответить на вопрос. Если ответа нет — скажи честно.\n\nКонтекст:\n{context}\n\nВопрос: {user_question}"
            print("🧪 GPT-поиск, т.к. точного совпадения нет...")
            completion = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400,
                temperature=0.3,
            )
            gpt_answer = completion.choices[0].message.content.strip()
            answer = f"🤖 Ответ GPT на основе базы знаний:\n{gpt_answer}"

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
