import os
import csv
import openai
from flask import Flask, request, jsonify
from flask_cors import CORS
from pathlib import Path

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Получаем ключ из переменной окружения
openai.api_key = os.getenv("OPENAI_API_KEY")

# Загрузка базы знаний
knowledge_base = []
try:
    kb_path = Path(__file__).resolve().parent / "knowledge_base.csv"
    with open(kb_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            knowledge_base.append({"question": row["question"], "answer": row["answer"]})
except Exception as e:
    print(f"❌ Ошибка при загрузке базы знаний: {e}")

# Поиск по базе знаний
def simple_search(user_question):
    return [
        item for item in knowledge_base
        if user_question.lower() in item["question"].lower()
    ][:3]

# Главный маршрут для чата
@app.route("/ask", methods=["POST", "OPTIONS"])
def ask():
    if request.method == "OPTIONS":
        return '', 200

    data = request.get_json()
    user_question = data.get("question", "")
    print("📥 Вопрос пользователя:", user_question)

    context_items = simple_search(user_question)
    context = "\n\n".join([f"Вопрос: {i['question']}\nОтвет: {i['answer']}" for i in context_items])
    print("📚 Контекст:", context)

    if context.strip():
        prompt = f"Ты эксперт по 44-ФЗ. Используй контекст ниже, чтобы ответить на вопрос:\n\n{context}\n\nВопрос: {user_question}"
    else:
        prompt = f"Ты эксперт по 44-ФЗ. Ответь на вопрос пользователя максимально полно:\n\nВопрос: {user_question}"

    try:
        print("🧪 Отправка запроса в OpenAI...")
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.7,
        )
        answer = response.choices[0].message.content
        return jsonify({"answer": answer})

    except Exception as e:
        print("❌ Ошибка OpenAI:", e)
        return jsonify({"error": str(e)}), 500

# Тестовый эндпоинт для проверки API-ключа
@app.route("/check-key", methods=["GET"])
def check_key():
    try:
        models = openai.models.list()
        model_names = [m.id for m in models.data]
        return jsonify({"status": "ok", "models": model_names})
    except Exception as e:
        print("❌ Ошибка при проверке ключа:", e)
        return jsonify({"status": "error", "message": str(e)}), 401

# Запуск сервера
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(debug=True, host='0.0.0.0', port=port)
