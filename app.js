document.getElementById('ask-question').addEventListener('click', async function() {
    const question = document.getElementById('user-question').value;
    const loader = document.getElementById('loader');
    const chatBox = document.getElementById('chat-box');

    if (question.trim() !== '') {
        chatBox.innerHTML += `<p><strong>Вы:</strong> ${question}</p>`;
        loader.style.display = 'block';

        try {
            const response = await fetch('https://ekspert-po-zakonu.onrender.com/ask', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ question })
            });

            const data = await response.json();
            console.log(data);
            chatBox.innerHTML += `<p><strong>Эксперт:</strong> ${data.answer}</p>`;
        } catch (error) {
            chatBox.innerHTML += `<p><strong>Ошибка:</strong> Не удалось получить ответ от сервера.</p>`;
        } finally {
            loader.style.display = 'none';
        }
    }
});
