const chatMessages = document.getElementById('chat-messages');
const userInput = document.getElementById('user-input');
const sendButton = document.getElementById('send-button');

const RASA_SERVER = 'http://localhost:5005';

async function sendToRasa(message) {
    try {
        const response = await fetch(`${RASA_SERVER}/webhooks/rest/webhook`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                sender: 'user',
                message: message
            })
        });

        const rasaResponse = await response.json();
        // Return the full array of responses instead of just the first one
        return rasaResponse;
    } catch (error) {
        console.error('Error:', error);
        return [{ text: 'Sorry, I am having trouble connecting to the server.' }];
    }
}

function addMessage(message, isUser) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${isUser ? 'user-message' : 'bot-message'}`;
    
    const messageContent = document.createElement('div');
    messageContent.className = 'message-content';
    messageContent.textContent = message;
    
    messageDiv.appendChild(messageContent);
    chatMessages.appendChild(messageDiv);
    
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

async function handleUserInput() {
    const message = userInput.value.trim();
    if (message) {
        addMessage(message, true);
        userInput.value = '';
        
        // Get responses from RASA
        const botResponses = await sendToRasa(message);
        
        // Handle multiple responses
        for (const response of botResponses) {
            if (response.text) {
                addMessage(response.text, false);
            }
            // Add a small delay between messages for better readability
            await new Promise(resolve => setTimeout(resolve, 500));
        }
    }
}

sendButton.addEventListener('click', handleUserInput);
userInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        handleUserInput();
    }
});