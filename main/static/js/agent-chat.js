/**
 * Agent Chat - Main Application
 */

class AgentChat {
    constructor() {
        this.messageInput = document.getElementById('messageInput');
        this.sendBtn = document.getElementById('sendBtn');
        this.newChatBtn = document.getElementById('newChatBtn');
        this.clearHistoryBtn = document.getElementById('clearHistoryBtn');
        this.messagesContainer = document.getElementById('messages');
        this.typingIndicator = document.getElementById('typingIndicator');
        this.currentModel = localStorage.getItem('selectedModel') || 'bloom';
        this.init();
    }

    init() {
        console.log('AgentChat initializing...');
        this.setupEventListeners();
        this.setupSuggestions();
        this.loadSettings();
        this.loadHistory();
        this.initModelSelector();
    }

    setupEventListeners() {
        if (this.sendBtn) {
            this.sendBtn.addEventListener('click', () => this.sendMessage());
        }

        if (this.messageInput) {
            this.messageInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    this.sendMessage();
                }
            });

            this.messageInput.addEventListener('input', () => {
                this.messageInput.style.height = 'auto';
                this.messageInput.style.height = Math.min(this.messageInput.scrollHeight, 100) + 'px';
            });
        }

        if (this.newChatBtn) {
            this.newChatBtn.addEventListener('click', () => this.newChat());
        }

        if (this.clearHistoryBtn) {
            this.clearHistoryBtn.addEventListener('click', () => this.clearHistory());
        }

        // Settings toggle
        const settingsBtn = document.getElementById('settingsToggleBtn');
        const settingsPanel = document.getElementById('settingsPanel');
        if (settingsBtn && settingsPanel) {
            settingsBtn.addEventListener('click', () => {
                settingsPanel.style.display = settingsPanel.style.display === 'none' ? 'block' : 'none';
            });
        }

        // Language button
        const langBtn = document.getElementById('langBtn');
        const langDropdown = document.getElementById('langDropdown');
        if (langBtn && langDropdown) {
            langBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                langDropdown.style.display = langDropdown.style.display === 'none' ? 'block' : 'none';
            });

            document.querySelectorAll('#langDropdown div').forEach(div => {
                div.addEventListener('click', () => {
                    const lang = div.dataset.lang;
                    if (window.langSwitcher) {
                        window.langSwitcher.setLanguage(lang);
                    }
                    langDropdown.style.display = 'none';
                });
            });

            document.addEventListener('click', () => {
                langDropdown.style.display = 'none';
            });
        }

        // Save settings
        const agentType = document.getElementById('agentTypeSelect');
        const citationStyle = document.getElementById('citationStyleSelect');
        const streamToggle = document.getElementById('streamToggle');
        const citationsToggle = document.getElementById('citationsToggle');
        const tempSlider = document.getElementById('temperatureSlider');
        const tempValue = document.getElementById('tempValue');

        if (agentType) {
            const saved = localStorage.getItem('agentType');
            if (saved) agentType.value = saved;
            agentType.addEventListener('change', () => {
                localStorage.setItem('agentType', agentType.value);
            });
        }

        if (citationStyle) {
            const saved = localStorage.getItem('citationStyle');
            if (saved) citationStyle.value = saved;
            citationStyle.addEventListener('change', () => {
                localStorage.setItem('citationStyle', citationStyle.value);
                if (window.citationManager) {
                    window.citationManager.setStyle(citationStyle.value);
                }
            });
        }

        if (tempSlider && tempValue) {
            const saved = localStorage.getItem('temperature');
            if (saved) tempSlider.value = saved;
            tempValue.textContent = tempSlider.value;
            tempSlider.addEventListener('input', (e) => {
                tempValue.textContent = e.target.value;
                localStorage.setItem('temperature', e.target.value);
            });
        }

        if (streamToggle) {
            streamToggle.checked = localStorage.getItem('streamResponse') !== 'false';
            streamToggle.addEventListener('change', () => {
                localStorage.setItem('streamResponse', streamToggle.checked);
            });
        }

        if (citationsToggle) {
            citationsToggle.checked = localStorage.getItem('showCitations') !== 'false';
            citationsToggle.addEventListener('change', () => {
                localStorage.setItem('showCitations', citationsToggle.checked);
            });
        }
    }

    initModelSelector() {
        const modelSelect = document.getElementById('modelSelect');
        if (modelSelect) {
            const saved = localStorage.getItem('selectedModel');
            if (saved) modelSelect.value = saved;
            modelSelect.addEventListener('change', () => {
                this.currentModel = modelSelect.value;
                localStorage.setItem('selectedModel', this.currentModel);
                this.showToast(`Switched to ${modelSelect.options[modelSelect.selectedIndex].text}`);
            });
        }
    }

    setupSuggestions() {
        document.querySelectorAll('.suggestion').forEach(btn => {
            btn.removeEventListener('click', this.suggestionHandler);
            this.suggestionHandler = () => {
                this.messageInput.value = btn.textContent;
                this.sendMessage();
            };
            btn.addEventListener('click', this.suggestionHandler);
        });
    }

    loadSettings() {
        // Settings loaded in event listeners
    }

    async sendMessage() {
        const message = this.messageInput.value.trim();
        if (!message) return;

        this.addMessage('user', message);
        this.messageInput.value = '';
        this.messageInput.style.height = 'auto';
        this.showTyping();

        try {
            const useStream = localStorage.getItem('streamResponse') !== 'false';
            const model = this.currentModel;
            const temperature = parseFloat(localStorage.getItem('temperature') || '0.7');
            const agentType = localStorage.getItem('agentType') || 'assistant';

            if (useStream && window.chatStreaming) {
                await window.chatStreaming.stream(message, model, temperature, agentType);
            } else {
                const response = await fetch('/api/chat/', {
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
                const data = await response.json();
                this.hideTyping();

                if (data.success) {
                    this.addMessage('assistant', data.response);
                    if (data.citations && data.citations.length > 0 && localStorage.getItem('showCitations') !== 'false') {
                        if (window.citationManager) {
                            window.citationManager.show(data.citations);
                        }
                    }
                } else {
                    this.addMessage('assistant', 'Sorry, an error occurred. Please try again.');
                }
            }
        } catch (error) {
            console.error('Error:', error);
            this.hideTyping();
            this.addMessage('assistant', 'Network error. Please check your connection.');
        }

        this.saveToHistory();
    }

    addMessage(role, content) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role}`;

        let formatted = content.replace(/\n/g, '<br>');
        formatted = formatted.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        formatted = formatted.replace(/`(.*?)`/g, '<code>$1</code>');

        messageDiv.innerHTML = `
            <div class="message-avatar">${role === 'user' ? 'You' : 'AI'}</div>
            <div class="message-content">
                <div class="message-text">${formatted}</div>
                <div class="message-time">${new Date().toLocaleTimeString()}</div>
                <div class="message-actions">
                    <button class="copy-msg">📋 Copy</button>
                    ${role === 'assistant' ? '<button class="speak-msg">🔊 Listen</button>' : ''}
                </div>
            </div>
        `;

        const copyBtn = messageDiv.querySelector('.copy-msg');
        if (copyBtn) {
            copyBtn.addEventListener('click', () => {
                navigator.clipboard.writeText(content);
                this.showToast('Copied!');
            });
        }

        const speakBtn = messageDiv.querySelector('.speak-msg');
        if (speakBtn) {
            speakBtn.addEventListener('click', () => {
                const utterance = new SpeechSynthesisUtterance(content);
                utterance.lang = 'en-US';
                utterance.rate = 1;
                window.speechSynthesis.cancel();
                window.speechSynthesis.speak(utterance);
            });
        }

        this.messagesContainer.appendChild(messageDiv);
        this.scrollToBottom();
    }

    showTyping() {
        if (this.typingIndicator) {
            this.typingIndicator.style.display = 'flex';
        }
        this.scrollToBottom();
    }

    hideTyping() {
        if (this.typingIndicator) {
            this.typingIndicator.style.display = 'none';
        }
    }

    scrollToBottom() {
        const container = document.getElementById('messagesContainer');
        if (container) {
            container.scrollTop = container.scrollHeight;
        }
    }

    saveToHistory() {
        const messages = [];
        document.querySelectorAll('#messages .message').forEach(msg => {
            const role = msg.classList.contains('user') ? 'user' : 'assistant';
            const text = msg.querySelector('.message-text').innerText;
            messages.push({ role, content: text });
        });

        if (messages.length === 0) return;

        const history = JSON.parse(localStorage.getItem('chatHistory') || '[]');
        history.unshift({
            id: Date.now(),
            title: messages[0]?.content.substring(0, 50) || 'New chat',
            messages: messages,
            timestamp: new Date().toISOString()
        });

        localStorage.setItem('chatHistory', JSON.stringify(history.slice(0, 50)));
        this.updateHistoryList();
    }

    loadHistory() {
        this.updateHistoryList();
    }

    updateHistoryList() {
        const list = document.getElementById('historyList');
        if (!list) return;

        const history = JSON.parse(localStorage.getItem('chatHistory') || '[]');

        if (history.length === 0) {
            list.innerHTML = '<li class="history-item empty-history">No chats yet</li>';
            return;
        }

        list.innerHTML = history.map(item => `
            <li class="history-item" data-id="${item.id}">
                <span>💬</span>
                <span>${this.escapeHtml(item.title)}</span>
                <span class="history-date">${this.formatDate(item.timestamp)}</span>
            </li>
        `).join('');

        list.querySelectorAll('.history-item[data-id]').forEach(item => {
            item.addEventListener('click', () => this.loadChat(item.dataset.id));
        });
    }

    loadChat(id) {
        const history = JSON.parse(localStorage.getItem('chatHistory') || '[]');
        const chat = history.find(h => h.id == id);

        if (chat && chat.messages) {
            this.messagesContainer.innerHTML = '';
            chat.messages.forEach(msg => {
                this.addMessage(msg.role, msg.content);
            });

            document.querySelectorAll('.history-item').forEach(item => {
                item.classList.remove('active');
                if (item.dataset.id == id) item.classList.add('active');
            });
        }
    }

    formatDate(timestamp) {
        const date = new Date(timestamp);
        const now = new Date();
        const diff = now - date;

        if (diff < 3600000) return 'Just now';
        if (diff < 86400000) return 'Today';
        if (diff < 172800000) return 'Yesterday';
        return date.toLocaleDateString();
    }

    newChat() {
        this.messagesContainer.innerHTML = '';
        const welcomeMsg = `
            <div class="message assistant">
                <div class="message-avatar">AI</div>
                <div class="message-content">
                    <div class="message-text">
                        <h3>👋 New conversation started!</h3>
                        <p>How can I help you with chronobiotics today?</p>
                        <div class="suggestions">
                            <button class="suggestion">What are chronobiotics?</button>
                            <button class="suggestion">Tell me about melatonin</button>
                            <button class="suggestion">How does KL001 work?</button>
                        </div>
                    </div>
                    <div class="message-time">Online</div>
                </div>
            </div>
        `;
        this.messagesContainer.innerHTML = welcomeMsg;
        this.setupSuggestions();
        this.saveToHistory();
    }

    clearHistory() {
        if (confirm('Clear all chat history?')) {
            localStorage.removeItem('chatHistory');
            this.newChat();
            this.updateHistoryList();
            this.showToast('History cleared!');
        }
    }

    showToast(message) {
        const toast = document.createElement('div');
        toast.textContent = message;
        toast.style.cssText = `
            position: fixed;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            background: #333;
            color: white;
            padding: 8px 16px;
            border-radius: 8px;
            z-index: 10000;
            font-size: 13px;
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

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.agentChat = new AgentChat();
});