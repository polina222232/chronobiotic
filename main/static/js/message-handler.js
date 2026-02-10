/**
 * Message Handler - Manages messages and storage
 */

class MessageHandler {
    constructor() {
        this.messages = [];
        this.loadMessages();
    }

    addMessage(role, content) {
        const message = {
            id: Date.now(),
            role: role,
            content: content,
            timestamp: new Date().toISOString()
        };
        this.messages.push(message);
        this.saveMessages();
        return message;
    }

    saveMessages() {
        localStorage.setItem('chatMessages', JSON.stringify(this.messages.slice(-50)));
    }

    loadMessages() {
        const saved = localStorage.getItem('chatMessages');
        if (saved) {
            this.messages = JSON.parse(saved);
            this.renderAll();
        }
    }

    renderAll() {
        const container = document.getElementById('messages');
        if (!container) return;

        container.innerHTML = '';
        this.messages.forEach(msg => {
            this.renderMessage(msg);
        });
        this.scrollToBottom();
    }

    renderMessage(message) {
        const container = document.getElementById('messages');
        const div = document.createElement('div');
        div.className = `message ${message.role}`;

        const formatted = window.markdownRenderer ? window.markdownRenderer.render(message.content) : message.content.replace(/\n/g, '<br>');

        div.innerHTML = `
            <div class="message-avatar">${message.role === 'user' ? 'You' : 'AI'}</div>
            <div class="message-content">
                <div class="message-text">${formatted}</div>
                <div class="message-time">${new Date(message.timestamp).toLocaleTimeString()}</div>
            </div>
        `;
        container.appendChild(div);
    }

    scrollToBottom() {
        const container = document.getElementById('messagesContainer');
        if (container) {
            container.scrollTop = container.scrollHeight;
        }
    }

    clear() {
        this.messages = [];
        localStorage.removeItem('chatMessages');
        const container = document.getElementById('messages');
        if (container) container.innerHTML = '';
    }
}

const messageHandler = new MessageHandler();
window.messageHandler = messageHandler;