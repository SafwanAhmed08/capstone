const chatMessages = document.getElementById('chat-messages');
const userInput = document.getElementById('user-input');
const sendButton = document.getElementById('send-button');

// Rasa server URL — update port to match the server you started (5006)
const RASA_SERVER = 'http://localhost:5006';

// Conversation sender id used by the UI and the inference notifier
const SENDER_ID = 'user';
// Tracker polling interval (ms)
const TRACKER_POLL_INTERVAL = 2000;
let lastTrackerEventCount = 0;
async function sendToRasa(message) {
    try {
        const response = await fetch(`${RASA_SERVER}/webhooks/rest/webhook`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                sender: SENDER_ID,
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

// Poll Rasa tracker to surface bot messages that were injected externally
async function pollTracker() {
    try {
        const resp = await fetch(`${RASA_SERVER}/conversations/${SENDER_ID}/tracker`);
        if (!resp.ok) {
            return;
        }
        const data = await resp.json();
        const events = data.events || [];

        // On first run, initialize the lastTrackerEventCount so we don't replay history
        if (lastTrackerEventCount === 0) {
            lastTrackerEventCount = events.length;
            return;
        }

        // Process new events only
        const newEvents = events.slice(lastTrackerEventCount);
        newEvents.forEach(ev => {
            if (ev.event === 'bot' && ev.text) {
                addMessage(ev.text, false);
            }
        });

        lastTrackerEventCount = events.length;
    } catch (e) {
        // silent fail - polling will retry
    }
}

// Start polling the tracker periodically so UI receives messages posted to Rasa by external tools
setInterval(pollTracker, TRACKER_POLL_INTERVAL);
// Run once on load to initialize
pollTracker();