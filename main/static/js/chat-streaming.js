/**
 * Chat Streaming - Real-time response streaming
 */

class ChatStreaming {
    constructor() {
        this.isStreaming = false;
        this.currentMessageId = null;
    }

    async stream(message, model = 'bloom', temperature = 0.7, agentType = 'assistant') {
        if (this.isStreaming) return;

        this.isStreaming = true;
        const typingIndicator = document.getElementById('typingIndicator');
        if (typingIndicator) typingIndicator.style.display = 'flex';

        const messageId = 'temp_' + Date.now();
        this.createTempMessage(messageId);

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
                })
            });

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
                                if (typingIndicator) typingIndicator.style.display = 'none';
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
        }

        this.isStreaming = false;
        if (typingIndicator) typingIndicator.style.display = 'none';
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
                <div class="message-text"><span class="spinner"></span> Thinking...</div>
                <div class="message-time">Typing...</div>
            </div>
        `;
        messagesDiv.appendChild(div);
        this.scrollToBottom();
    }

    updateTempMessage(id, text) {
        const div = document.getElementById(id);
        if (div) {
            let formatted = text.replace(/\n/g, '<br>');
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
            actions.innerHTML = '<button class="copy-msg">📋 Copy</button><button class="speak-msg">🔊 Listen</button>';

            const copyBtn = actions.querySelector('.copy-msg');
            copyBtn.addEventListener('click', () => {
                navigator.clipboard.writeText(text);
            });

            const speakBtn = actions.querySelector('.speak-msg');
            speakBtn.addEventListener('click', () => {
                const utterance = new SpeechSynthesisUtterance(text);
                utterance.lang = 'en-US';
                utterance.rate = 1;
                window.speechSynthesis.cancel();
                window.speechSynthesis.speak(utterance);
            });

            div.querySelector('.message-content').appendChild(actions);

            // Show citations
            if (citations && citations.length > 0 && localStorage.getItem('showCitations') !== 'false') {
                if (window.citationManager) {
                    window.citationManager.show(citations);
                }
            }
        }
    }

    scrollToBottom() {
        const container = document.getElementById('messagesContainer');
        if (container) {
            container.scrollTop = container.scrollHeight;
        }
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
}

const chatStreaming = new ChatStreaming();
window.chatStreaming = chatStreaming;