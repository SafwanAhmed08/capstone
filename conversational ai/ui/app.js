const chatMessages = document.getElementById('chat-messages');
const userInput = document.getElementById('user-input');
const sendButton = document.getElementById('send-button');

// Rasa server URL — update port to match the server you started (5006)
const RASA_SERVER = 'http://localhost:5006';

// Conversation sender id used by the UI and the inference notifier
const SENDER_ID = 'user';
// Tracker polling interval (ms)
const TRACKER_POLL_INTERVAL = 2000;
// Feature flags for troubleshooting
const ENABLE_TRACKER_POLL = true; // enabled to surface external notifications injected into the tracker
const DEBUG_DUPES = true; // logs sources of messages
// When the UI itself sends a message via REST, suppress showing tracker events for the same turn
let suppressTrackerUntil = 0; // timestamp (ms) until which tracker events are hidden
let lastTrackerEventCount = 0;

// Dedupe recent bot messages so the same text isn't rendered twice when
// it comes via immediate REST webhook response and then via tracker polling.
// Map of message text -> last seen timestamp (ms)
const displayedMessages = new Map();
const DEDUPE_TTL_MS = 5000; // consider duplicates within 5 seconds as same

function shouldDisplayBotText(text) {
    if (!text) return false;
    // Normalize to avoid duplicates that differ only in spacing/case
    const key = text.trim().replace(/\s+/g, ' ').toLowerCase();
    const now = Date.now();
    const last = displayedMessages.get(key);
    if (last && (now - last) < DEDUPE_TTL_MS) {
        // Recently shown; skip duplicate
        return false;
    }
    displayedMessages.set(key, now);
    // Simple pruning to avoid unbounded growth
    if (displayedMessages.size > 1000) {
        // Remove oldest ~25% entries
        const entries = Array.from(displayedMessages.entries()).sort((a, b) => a[1] - b[1]);
        const toRemove = Math.ceil(entries.length * 0.25);
        for (let i = 0; i < toRemove; i++) {
            displayedMessages.delete(entries[i][0]);
        }
    }
    return true;
}
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

function addMessage(message, isUser, source = 'UNKNOWN') {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${isUser ? 'user-message' : 'bot-message'}`;
    
    const messageContent = document.createElement('div');
    messageContent.className = 'message-content';
    messageContent.textContent = message;

    if (!isUser && DEBUG_DUPES) {
        const metaSpan = document.createElement('span');
        metaSpan.style.display = 'block';
        metaSpan.style.fontSize = '0.6rem';
        metaSpan.style.opacity = '0.5';
        metaSpan.textContent = `[${source}]`;
        messageDiv.appendChild(metaSpan);
        // Also log to console
        console.log(`Bot message (${source}):`, message);
    }
    
    messageDiv.appendChild(messageContent);
    chatMessages.appendChild(messageDiv);
    
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

async function handleUserInput() {
    const message = userInput.value.trim();
    if (message) {
        addMessage(message, true);
        userInput.value = '';
        // Prevent double send while awaiting response
        sendButton.disabled = true;
        userInput.disabled = true;
    // Set suppression window so tracker-poll doesn't mirror REST responses for this turn
    suppressTrackerUntil = Date.now() + 10000; // 10s suppression window
        
        // Get responses from RASA
        const botResponses = await sendToRasa(message);
        
        // Handle multiple responses
        for (const response of botResponses) {
            if (response.text && shouldDisplayBotText(response.text)) {
                addMessage(response.text, false, 'REST');
            }
            // Add a small delay between messages for better readability
            await new Promise(resolve => setTimeout(resolve, 500));
        }

        // Extend suppression slightly after rendering REST responses
        suppressTrackerUntil = Date.now() + 10000;

        // Re-enable input
        sendButton.disabled = false;
        userInput.disabled = false;
        userInput.focus();
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

        // If we're within the suppression window of a recent UI-originated REST send,
        // advance the event cursor but do not render tracker bot events for this turn
        if (Date.now() <= suppressTrackerUntil) {
            lastTrackerEventCount = events.length;
            return;
        }

        // Process new events only
        const newEvents = events.slice(lastTrackerEventCount);
        newEvents.forEach(ev => {
            if (ev.event === 'bot' && ev.text) {
                // Skip showing tracker echoes right after a UI-originated REST send
                if (Date.now() <= suppressTrackerUntil) {
                    return;
                }
                if (shouldDisplayBotText(ev.text)) {
                    addMessage(ev.text, false, 'TRACKER');
                }
            }
        });

        lastTrackerEventCount = events.length;
    } catch (e) {
        // silent fail - polling will retry
    }
}

// Start polling the tracker periodically so UI receives messages posted to Rasa by external tools
if (ENABLE_TRACKER_POLL) {
    setInterval(pollTracker, TRACKER_POLL_INTERVAL);
    // Run once on load to initialize
    pollTracker();
}