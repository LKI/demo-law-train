const chatContainer = document.getElementById('chatContainer');
const userInput = document.getElementById('userInput');
const sendBtn = document.getElementById('sendBtn');

const STORAGE_KEY = 'chat_law_demo_history_v1';
let chatHistory = [];

// Load history on startup
loadHistory();

// Auto-resize textarea
userInput.addEventListener('input', function () {
    this.style.height = 'auto';
    this.style.height = (this.scrollHeight) + 'px';
    sendBtn.disabled = this.value.trim().length === 0;
});

// Handle Enter key to submit
userInput.addEventListener('keydown', function (e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        if (this.value.trim().length > 0) {
            sendMessage();
        }
    }
});

sendBtn.addEventListener('click', sendMessage);

function saveHistory() {
    try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(chatHistory));
    } catch (e) {
        console.error('Failed to save history', e);
    }
}

function loadHistory() {
    try {
        const saved = localStorage.getItem(STORAGE_KEY);
        if (saved) {
            chatHistory = JSON.parse(saved);
            chatHistory.forEach(item => {
                if (item.type === 'user') {
                    appendMessage(item.content, 'user');
                } else if (item.type === 'comparison') {
                    const comp = appendComparisonBlock();
                    comp.baseContent.textContent = item.base || '...';
                    comp.loraContent.textContent = item.lora || '...';
                    // Remove typing class for loaded messages
                    comp.baseContent.classList.remove('typing');
                    comp.loraContent.classList.remove('typing');
                    // Reset text content if it was empty/dots to something valid if needed, 
                    // but '...' is fine for now if empty.
                }
            });
            scrollToBottom();
        }
    } catch (e) {
        console.error('Failed to load history', e);
        // If corrupt, clear it
        localStorage.removeItem(STORAGE_KEY);
        chatHistory = [];
    }
}

async function sendMessage() {
    const text = userInput.value.trim();
    if (!text) return;

    // Reset input
    userInput.value = '';
    userInput.style.height = 'auto';
    sendBtn.disabled = true;

    // Add User Message
    appendMessage(text, 'user');

    // Save User Message immediately
    chatHistory.push({ type: 'user', content: text, timestamp: Date.now() });
    saveHistory();

    // Add comparison placeholders
    const comparison = appendComparisonBlock();
    let baseText = '';
    let loraText = '';

    try {
        const response = await fetch('/api/compare', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ message: text }),
        });

        if (!response.ok || !response.body) {
            throw new Error('Network response was not ok');
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let baseDone = false;
        let loraDone = false;

        while (true) {
            const { done, value } = await reader.read();
            buffer += decoder.decode(value || new Uint8Array(), { stream: !done });

            const lines = buffer.split('\n');
            buffer = lines.pop() ?? '';

            for (const line of lines) {
                if (!line.trim()) continue;
                let data;
                try {
                    data = JSON.parse(line);
                } catch (err) {
                    console.warn('Failed to parse chunk', err, line);
                    continue;
                }

                const { model, delta, done: modelDone } = data;
                if (model === 'base') {
                    if (delta) baseText += delta;
                    comparison.baseContent.textContent = baseText || '...';
                    if (modelDone) {
                        baseDone = true;
                        comparison.baseContent.classList.remove('typing');
                    }
                } else if (model === 'lora') {
                    if (delta) loraText += delta;
                    comparison.loraContent.textContent = loraText || '...';
                    if (modelDone) {
                        loraDone = true;
                        comparison.loraContent.classList.remove('typing');
                    }
                }
                scrollToBottom();
            }

            if (done) {
                // Handle any trailing buffer
                if (buffer.trim()) {
                    try {
                        const data = JSON.parse(buffer);
                        const { model, delta, done: modelDone } = data;
                        if (model === 'base') {
                            if (delta) baseText += delta;
                            comparison.baseContent.textContent = baseText || '...';
                            if (modelDone) comparison.baseContent.classList.remove('typing');
                        } else if (model === 'lora') {
                            if (delta) loraText += delta;
                            comparison.loraContent.textContent = loraText || '...';
                            if (modelDone) comparison.loraContent.classList.remove('typing');
                        }
                    } catch (err) {
                        console.warn('Failed to parse trailing chunk', err, buffer);
                    }
                }
                break;
            }
        }

        // Save Completed Response (Base + LoRA)
        chatHistory.push({
            type: 'comparison',
            base: baseText,
            lora: loraText,
            timestamp: Date.now()
        });
        saveHistory();

    } catch (error) {
        console.error('Error:', error);
        comparison.baseContent.textContent = 'Error fetching base model response.';
        comparison.loraContent.textContent = 'Error fetching LoRA model response.';
        comparison.baseContent.classList.remove('typing');
        comparison.loraContent.classList.remove('typing');
        comparison.baseContent.style.color = '#ef4444';
        comparison.loraContent.style.color = '#ef4444';
    } finally {
        sendBtn.disabled = false;
    }
}

function appendMessage(text, sender) {
    const msgDiv = document.createElement('div');
    msgDiv.classList.add('message', `${sender}-message`);

    const contentDiv = document.createElement('div');
    contentDiv.classList.add('message-content');
    contentDiv.textContent = text;

    msgDiv.appendChild(contentDiv);
    chatContainer.appendChild(msgDiv);
    scrollToBottom();

    return contentDiv;
}

function appendComparisonBlock() {
    const wrapper = document.createElement('div');
    wrapper.classList.add('comparison-block');

    const baseCard = createModelCard('Base Model', 'base');
    const loraCard = createModelCard('LoRA Model', 'lora');

    wrapper.appendChild(baseCard.card);
    wrapper.appendChild(loraCard.card);
    chatContainer.appendChild(wrapper);
    scrollToBottom();

    return {
        baseContent: baseCard.content,
        loraContent: loraCard.content,
    };
}

function createModelCard(label, type) {
    const card = document.createElement('div');
    card.classList.add('model-card');

    const header = document.createElement('div');
    header.classList.add('model-header');

    const badge = document.createElement('span');
    badge.classList.add('badge', type);
    badge.textContent = type === 'base' ? 'BASE' : 'LORA';

    const title = document.createElement('span');
    title.textContent = label;

    header.appendChild(badge);
    header.appendChild(title);

    const content = document.createElement('div');
    content.classList.add('model-content', 'typing');
    content.textContent = '正在生成...';

    card.appendChild(header);
    card.appendChild(content);

    return { card, content };
}

function scrollToBottom() {
    chatContainer.scrollTop = chatContainer.scrollHeight;
}
