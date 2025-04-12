document.getElementById('ask-question').addEventListener('click', async function() {
    const question = document.getElementById('user-question').value;
    if (question.trim() !== '') {
        const chatBox = document.getElementById('chat-box');
        chatBox.innerHTML += `<p><strong>Вы:</strong> ${question}</p>`;

        // Запрос к бэкенду на Render
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
    }
});
