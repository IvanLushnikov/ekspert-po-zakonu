import os
import openai
from flask import Flask, request, jsonify
from flask_cors import CORS  # добавляем импорт CORS

# --- Настройки Flask и OpenAI ---
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "https://ekspert-po-zakonu.vercel.app"}})  # Разрешаем запросы с этого домена
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

# --- Маршрут для GPT-чата ---
@app.route("/ask", methods=["POST"])
def ask():
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
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        answer = response.choices[0].message["content"]
        return jsonify({"answer": answer})
    except Exception as e:
        print("❌ Ошибка OpenAI:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
