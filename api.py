import openai
import os
from flask import Flask, request, jsonify
from flask_cors import CORS  

app = Flask(__name__)
CORS(app, origins=["https://ekspert-po-zakonu.vercel.app"]) 

# Загружаем текст законов из файла
with open("all_laws_combined.txt", encoding="utf-8") as f:
    LAWS_TEXT = f.read()

openai.api_key = os.getenv("OPENAI_API_KEY")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    user_question = data.get("question")

    if not user_question:
        return jsonify({"error": "No question provided"}), 400

    prompt = f"""Контекст (законодательство):\n{LAWS_TEXT[:120000]}\n\nВопрос: {user_question}"""

    messages = [
        {"role": "system", "content": "Ты — эксперт по государственным закупкам РФ. Отвечай строго по закону, но простыми словами. Ссылайся на статьи, если они есть."},
        {"role": "user", "content": prompt}
    ]

    try:
        response = openai.ChatCompletion.create(
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
