import os
import csv
import openai
from flask import Flask, request, jsonify
from flask_cors import CORS  # Подключаем CORS
from pathlib import Path

# --- Настройки Flask и OpenAI ---
app = Flask(__name__)

# Разрешаем CORS-запросы
CORS(app, resources={r"/*": {"origins": "*"}})

# Получаем ключ API из переменных окружения
openai.api_key = os.getenv("OPENAI_API_KEY")

# --- Загрузка базы знаний из CSV ---
knowledge_base = []
try:
    kb_path = Path(__file__).resolve().parent / "knowledge_base.csv"
    with open(kb_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            knowledge_base.append({"question": row["question"], "answer": row["answer"]})
except Exception as e:
    print(f"Ошибка при загрузке базы знаний: {e}")

# --- Поиск по базе ---
def simple_search(user_question):
    results = []
    for item in knowledge_base:
        if user_question.lower() in item["question"].lower():
            results.append(item)
    return results[:3]

# --- Маршрут для обработки запросов GPT-чата ---
@app.route("/ask", methods=["POST", "OPTIONS"])
def ask():
    # Обработка метода OPTIONS для CORS
    if request.method == "OPTIONS":
        return '', 200  # Ответ на запрос OPTIONS для CORS

    # Обработка POST-запроса
    data = request.get_json()
    user_question = data.get("question", "")
    print("📥 Вопрос пользователя:", user_question)

    # Поиск в базе знаний
    context_items = simple_search(user_question)
    context = "\n\n".join([f"Вопрос: {i['question']}\nОтвет: {i['answer']}" for i in context_items])
    print("📚 Контекст:", context)

    if context.strip():
        # Если контекст найден, используем его для построения запроса к GPT
        prompt = f"Ты эксперт по 44-ФЗ. Используй контекст ниже, чтобы ответить на вопрос:\n\n{context}\n\nВопрос: {user_question}"
    else:
        # Если контекст не найден, задаем вопрос напрямую
        prompt = f"Ты эксперт по 44-ФЗ. Ответь на вопрос пользователя максимально полно:\n\nВопрос: {user_question}"

    try:
        # Отправка запроса в OpenAI
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        # Получаем ответ и отправляем его клиенту
        answer = response.choices[0].message["content"]
        print("Ответ от OpenAI:", answer)  # Добавьте эту строку для логирования ответа
        return jsonify({"answer": answer})
    except Exception as e:
        # Обработка ошибок, если запрос не удался
        print("❌ Ошибка OpenAI:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Получаем порт из окружения
    app.run(debug=True, host='0.0.0.0', port=port)
