/**
 * Chat Streaming - Real-time response streaming
 */

class ChatStreaming {
    constructor() {
        this.isStreaming = false;
        this.currentMessageId = null;
        this.abortController = null;
    }

    async stream(message, model = 'bloom', temperature = 0.7, agentType = 'assistant') {
        if (this.isStreaming) {
            this.stop();
            return;
        }

        this.isStreaming = true;
        this.showTypingIndicator();

        const messageId = 'temp_' + Date.now();
        this.createTempMessage(messageId);

        this.abortController = new AbortController();

        try {
            const response = await fetch('/api/chat/stream/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': this.getCookie('csrftoken')
                },
                body: JSON.stringify({
                    message: message,
                    model: model,
                    temperature: temperature,
                    agent_type: agentType
                }),
                signal: this.abortController.signal
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let fullText = '';
            let buffer = '';

            while (this.isStreaming) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop() || '';

                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        try {
                            const data = JSON.parse(line.slice(6));
                            if (data.chunk) {
                                fullText += data.chunk;
                                this.updateTempMessage(messageId, fullText);
                            }
                            if (data.done) {
                                this.finalizeMessage(messageId, fullText, data.citations);
                                this.isStreaming = false;
                                this.hideTypingIndicator();
                                return;
                            }
                        } catch (e) {
                            console.error('Parse error:', e);
                        }
                    }
                }
            }

            this.finalizeMessage(messageId, fullText);
        } catch (error) {
            console.error('Streaming error:', error);
            this.updateTempMessage(messageId, 'Sorry, an error occurred. Please try again.');
            this.showToast('Connection error', '#dc3545');
        }

        this.isStreaming = false;
        this.hideTypingIndicator();
    }

    createTempMessage(id) {
        const messagesDiv = document.getElementById('messages');
        if (!messagesDiv) return;

        const div = document.createElement('div');
        div.className = 'message assistant temp';
        div.id = id;
        div.innerHTML = `
            <div class="message-avatar">AI</div>
            <div class="message-content">
                <div class="message-text">
                    <div class="typing-dots">
                        <span></span><span></span><span></span>
                    </div>
                </div>
                <div class="message-time">Typing...</div>
            </div>
        `;
        messagesDiv.appendChild(div);
        this.scrollToBottom();
    }

    updateTempMessage(id, text) {
        const div = document.getElementById(id);
        if (div) {
            let formatted = this.escapeHtml(text);
            formatted = formatted.replace(/\n/g, '<br>');
            formatted = formatted.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
            div.querySelector('.message-text').innerHTML = formatted;
            this.scrollToBottom();
        }
    }

    finalizeMessage(id, text, citations = null) {
        const div = document.getElementById(id);
        if (div) {
            div.classList.remove('temp');
            div.querySelector('.message-time').textContent = new Date().toLocaleTimeString();
            div.id = '';

            const actions = document.createElement('div');
            actions.className = 'message-actions';
            actions.innerHTML = `
                <button class="copy-msg" title="Copy">📋 Copy</button>
                <button class="speak-msg" title="Listen">🔊 Listen</button>
                <button class="regenerate-msg" title="Regenerate">🔄 Regenerate</button>
            `;

            const copyBtn = actions.querySelector('.copy-msg');
            copyBtn.addEventListener('click', () => {
                navigator.clipboard.writeText(text);
                this.showToast('Copied!', '#28a745');
            });

            const speakBtn = actions.querySelector('.speak-msg');
            speakBtn.addEventListener('click', () => {
                if (window.voicePlayer) {
                    window.voicePlayer.speak(text);
                }
            });

            const regenerateBtn = actions.querySelector('.regenerate-msg');
            regenerateBtn.addEventListener('click', () => {
                if (window.agentChat) {
                    window.agentChat.regenerateMessage(div.dataset.messageId);
                }
            });

            div.querySelector('.message-content').appendChild(actions);

            if (citations && citations.length > 0 && localStorage.getItem('showCitations') !== 'false') {
                if (window.citationManager) {
                    window.citationManager.show(citations);
                }
            }
        }
    }

    stop() {
        if (this.abortController) {
            this.abortController.abort();
            this.abortController = null;
        }
        this.isStreaming = false;
        this.hideTypingIndicator();
    }

    showTypingIndicator() {
        const indicator = document.getElementById('typingIndicator');
        if (indicator) indicator.style.display = 'flex';
    }

    hideTypingIndicator() {
        const indicator = document.getElementById('typingIndicator');
        if (indicator) indicator.style.display = 'none';
    }

    scrollToBottom() {
        const container = document.getElementById('messagesContainer');
        if (container && localStorage.getItem('autoScroll') !== 'false') {
            container.scrollTop = container.scrollHeight;
        }
    }

    showToast(message, bg = '#28a745') {
        const toast = document.createElement('div');
        toast.textContent = message;
        toast.style.cssText = `
            position: fixed;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            background: ${bg};
            color: white;
            padding: 6px 12px;
            border-radius: 6px;
            z-index: 10001;
            font-size: 12px;
        `;
        document.body.appendChild(toast);
        setTimeout(() => toast.remove(), 2000);
    }

    getCookie(name) {
        let value = null;
        if (document.cookie && document.cookie !== '') {
            const cookies = document.cookie.split(';');
            for (let i = 0; i < cookies.length; i++) {
                const cookie = cookies[i].trim();
                if (cookie.substring(0, name.length + 1) === (name + '=')) {
                    value = decodeURIComponent(cookie.substring(name.length + 1));
                    break;
                }
            }
        }
        return value;
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

const chatStreaming = new ChatStreaming();
window.chatStreaming = chatStreaming;