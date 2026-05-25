/**
 * Agent Chat - Main Application
 */

class AgentChat {
    constructor() {
        this.messageInput = null;
        this.sendBtn = null;
        this.newChatBtn = null;
        this.clearHistoryBtn = null;
        this.exportChatBtn = null;
        this.messagesContainer = null;
        this.typingIndicator = null;
        this.editPanel = null;
        this.editInput = null;
        this.saveEditBtn = null;
        this.cancelEditBtn = null;
        this.currentEditId = null;
        this.currentEditOriginal = null;
        this.isLoading = false;
        this.initialized = false;
    }

    init() {
        if (this.initialized) return;
        
        console.log('AgentChat initializing...');
        
        // Get DOM elements
        this.messageInput = document.getElementById('messageInput');
        this.sendBtn = document.getElementById('sendBtn');
        this.newChatBtn = document.getElementById('newChatBtn');
        this.clearHistoryBtn = document.getElementById('clearHistoryBtn');
        this.exportChatBtn = document.getElementById('exportChatBtn');
        this.messagesContainer = document.getElementById('messages');
        this.typingIndicator = document.getElementById('typingIndicator');
        this.editPanel = document.getElementById('editPanel');
        this.editInput = document.getElementById('editInput');
        this.saveEditBtn = document.getElementById('saveEditBtn');
        this.cancelEditBtn = document.getElementById('cancelEditBtn');
        
        this.setupEventListeners();
        this.loadHistory();
        this.setupSuggestions();
        this.initMobileMenu();
        this.loadSettings();
        this.initialized = true;
        console.log('AgentChat initialized successfully');
    }

    setupEventListeners() {
        if (this.sendBtn) {
            this.sendBtn.addEventListener('click', () => this.sendMessage());
        }

        if (this.newChatBtn) {
            this.newChatBtn.addEventListener('click', () => this.newChat());
        }

        if (this.clearHistoryBtn) {
            this.clearHistoryBtn.addEventListener('click', () => this.clearHistory());
        }

        if (this.exportChatBtn) {
            this.exportChatBtn.addEventListener('click', () => this.exportChat());
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
                this.messageInput.style.height = Math.min(this.messageInput.scrollHeight, 120) + 'px';
            });
        }

        if (this.saveEditBtn) {
            this.saveEditBtn.addEventListener('click', () => this.saveEdit());
        }

        if (this.cancelEditBtn) {
            this.cancelEditBtn.addEventListener('click', () => this.hideEditPanel());
        }

        const modelSelect = document.getElementById('modelSelect');
        if (modelSelect) {
            const savedModel = localStorage.getItem('selectedModel');
            if (savedModel) modelSelect.value = savedModel;
            modelSelect.addEventListener('change', () => {
                localStorage.setItem('selectedModel', modelSelect.value);
                this.showToast(`Switched to ${modelSelect.options[modelSelect.selectedIndex].text}`);
            });
        }
    }

    initMobileMenu() {
        const mobileBtn = document.getElementById('mobileMenuBtn');
        const sidebar = document.getElementById('chatSidebar');
        if (mobileBtn && sidebar) {
            mobileBtn.addEventListener('click', () => {
                sidebar.classList.toggle('open');
            });
        }
    }

    setupSuggestions() {
        // Use event delegation for better performance
        const suggestionsContainer = document.querySelector('.suggestions');
        if (suggestionsContainer) {
            suggestionsContainer.addEventListener('click', (e) => {
                const btn = e.target.closest('.suggestion');
                if (btn && this.messageInput) {
                    this.messageInput.value = btn.textContent;
                    this.sendMessage();
                }
            });
        } else {
            // Fallback to individual buttons
            document.querySelectorAll('.suggestion').forEach(btn => {
                btn.addEventListener('click', () => {
                    if (this.messageInput) {
                        this.messageInput.value = btn.textContent;
                        this.sendMessage();
                    }
                });
            });
        }
    }

    loadSettings() {
        const streamToggle = document.getElementById('streamToggle');
        if (streamToggle) {
            streamToggle.checked = localStorage.getItem('streamResponse') !== 'false';
        }

        const citationsToggle = document.getElementById('citationsToggle');
        if (citationsToggle) {
            citationsToggle.checked = localStorage.getItem('showCitations') !== 'false';
        }
    }

    async sendMessage() {
        if (this.isLoading) return;

        const message = this.messageInput.value.trim();
        if (!message) return;

        this.addMessage('user', message);
        this.messageInput.value = '';
        this.messageInput.style.height = 'auto';
        this.showTyping();
        this.isLoading = true;

        try {
            const useStream = localStorage.getItem('streamResponse') !== 'false';

            if (useStream && window.chatStreaming) {
                await window.chatStreaming.stream(message);
            } else {
                const response = await this.callAPI(message);
                this.addMessage('assistant', response);
            }
        } catch (error) {
            console.error('Send message error:', error);
            // Fallback response when API is not available
            const fallbackResponse = this.getFallbackResponse(message);
            this.addMessage('assistant', fallbackResponse);
        }

        this.hideTyping();
        this.isLoading = false;
        this.saveToHistory();
        this.updateHistoryList();
    }

    getFallbackResponse(message) {
        const msg = message.toLowerCase();
        if (msg.includes('chronobiotic')) {
            return "**Chronobiotics** are pharmacological agents that modify circadian rhythm parameters. They include natural compounds like melatonin, synthetic modulators like KL001 and KS15.";
        } else if (msg.includes('melatonin')) {
            return "**Melatonin** is a hormone produced by the pineal gland that regulates the sleep-wake cycle. It acts on MT1 and MT2 receptors.";
        } else {
            return "I'm ChronobioticsAI! I can help with chronobiotics, circadian rhythms, and research. Try asking about melatonin, KL001, or chronobiotics classification.";
        }
    }

    async callAPI(message) {
        try {
            const response = await fetch('/api/chat/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': this.getCookie('csrftoken')
                },
                body: JSON.stringify({ message: message })
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }

            const data = await response.json();

            if (data.citations && data.citations.length > 0 && localStorage.getItem('showCitations') !== 'false') {
                if (window.citationManager) {
                    window.citationManager.show(data.citations);
                }
            }

            return data.response || this.getFallbackResponse(message);
        } catch (error) {
            console.error('API call error:', error);
            return this.getFallbackResponse(message);
        }
    }

    addMessage(role, content, customId = null) {
        const messageId = customId || Date.now().toString();
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role}`;
        messageDiv.dataset.messageId = messageId;

        let formatted = this.escapeHtml(content);
        formatted = formatted.replace(/\n/g, '<br>');
        formatted = formatted.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        formatted = formatted.replace(/`(.*?)`/g, '<code>$1</code>');

        const avatar = role === 'user' ? 'You' : 'AI';
        const time = new Date().toLocaleTimeString();

        messageDiv.innerHTML = `
            <div class="message-avatar">${avatar}</div>
            <div class="message-content">
                <div class="message-text">${formatted}</div>
                <div class="message-time">${time}</div>
                <div class="message-actions">
                    <button class="copy-msg" title="Copy">📋 Copy</button>
                    ${role === 'user' ? '<button class="edit-msg" title="Edit">✏️ Edit</button>' : ''}
                    ${role === 'assistant' ? '<button class="speak-msg" title="Listen">🔊 Listen</button><button class="regenerate-msg" title="Regenerate">🔄 Regenerate</button>' : ''}
                </div>
            </div>
        `;

        this.addMessageEventListeners(messageDiv, role, content);
        this.messagesContainer.appendChild(messageDiv);

        if (localStorage.getItem('autoScroll') !== 'false') {
            this.scrollToBottom();
        }

        return messageId;
    }

    addMessageEventListeners(messageDiv, role, content) {
        const copyBtn = messageDiv.querySelector('.copy-msg');
        if (copyBtn) {
            copyBtn.addEventListener('click', () => {
                navigator.clipboard.writeText(content);
                this.showToast('Copied!');
            });
        }

        if (role === 'user') {
            const editBtn = messageDiv.querySelector('.edit-msg');
            if (editBtn) {
                editBtn.addEventListener('click', () => {
                    this.showEditPanel(messageDiv.dataset.messageId, content);
                });
            }
        }

        if (role === 'assistant') {
            const speakBtn = messageDiv.querySelector('.speak-msg');
            if (speakBtn && window.voicePlayer) {
                speakBtn.addEventListener('click', () => {
                    window.voicePlayer.speak(content);
                });
            }

            const regenerateBtn = messageDiv.querySelector('.regenerate-msg');
            if (regenerateBtn) {
                regenerateBtn.addEventListener('click', () => {
                    this.regenerateMessage(messageDiv.dataset.messageId);
                });
            }
        }
    }

    showEditPanel(messageId, content) {
        this.currentEditId = messageId;
        this.currentEditOriginal = content;
        if (this.editInput) this.editInput.value = content;
        if (this.editPanel) this.editPanel.style.display = 'flex';
        this.editInput?.focus();
    }

    hideEditPanel() {
        if (this.editPanel) this.editPanel.style.display = 'none';
        this.currentEditId = null;
        this.currentEditOriginal = null;
        if (this.editInput) this.editInput.value = '';
    }

    saveEdit() {
        const newContent = this.editInput?.value.trim();
        if (!newContent || !this.currentEditId) return;

        const messageDiv = this.messagesContainer.querySelector(`[data-message-id="${this.currentEditId}"]`);
        if (messageDiv) {
            const textDiv = messageDiv.querySelector('.message-text');
            if (textDiv) {
                textDiv.innerHTML = this.formatText(newContent);
            }

            let nextSibling = messageDiv.nextSibling;
            while (nextSibling) {
                const toRemove = nextSibling;
                nextSibling = nextSibling.nextSibling;
                if (toRemove.classList && toRemove.classList.contains('message')) {
                    toRemove.remove();
                }
            }

            this.hideEditPanel();
            this.saveToHistory();
            this.messageInput.value = newContent;
            this.sendMessage();
        }
    }

    regenerateMessage(messageId) {
        const messageDiv = this.messagesContainer.querySelector(`[data-message-id="${messageId}"]`);
        if (messageDiv) {
            let prev = messageDiv.previousSibling;
            let userMessage = null;
            while (prev) {
                if (prev.classList && prev.classList.contains('user')) {
                    userMessage = prev;
                    break;
                }
                prev = prev.previousSibling;
            }

            if (userMessage) {
                const userText = userMessage.querySelector('.message-text').innerText;
                let next = messageDiv;
                while (next) {
                    const toRemove = next;
                    next = next.nextSibling;
                    if (toRemove.classList && toRemove.classList.contains('message')) {
                        toRemove.remove();
                    }
                }
                this.messageInput.value = userText;
                this.sendMessage();
            }
        }
    }

    formatText(text) {
        let formatted = this.escapeHtml(text);
        formatted = formatted.replace(/\n/g, '<br>');
        formatted = formatted.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        formatted = formatted.replace(/`(.*?)`/g, '<code>$1</code>');
        return formatted;
    }

    showTyping() {
        if (this.typingIndicator) {
            this.typingIndicator.style.display = 'flex';
        }
        if (localStorage.getItem('autoScroll') !== 'false') {
            this.scrollToBottom();
        }
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
            const content = msg.querySelector('.message-text').innerText;
            messages.push({ role, content, id: msg.dataset.messageId });
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
            list.innerHTML = '<li class="history-item empty">No chats yet</li>';
            return;
        }

        list.innerHTML = history.map(item => `
            <li class="history-item" data-id="${item.id}">
                <span>💬</span>
                <span class="history-title">${this.escapeHtml(item.title)}</span>
                <span class="history-date">${this.formatDate(item.timestamp)}</span>
            </li>
        `).join('');

        list.querySelectorAll('.history-item[data-id]').forEach(item => {
            item.addEventListener('click', () => this.loadChat(item.dataset.id));
        });

        const searchInput = document.getElementById('historySearch');
        if (searchInput) {
            searchInput.addEventListener('input', (e) => {
                const query = e.target.value.toLowerCase();
                const items = list.querySelectorAll('.history-item[data-id]');
                items.forEach(item => {
                    const title = item.querySelector('.history-title')?.innerText.toLowerCase() || '';
                    if (title.includes(query)) {
                        item.style.display = 'flex';
                    } else {
                        item.style.display = 'none';
                    }
                });
            });
        }
    }

    loadChat(id) {
        const history = JSON.parse(localStorage.getItem('chatHistory') || '[]');
        const chat = history.find(h => h.id == id);

        if (chat && chat.messages) {
            this.messagesContainer.innerHTML = '';
            chat.messages.forEach(msg => {
                this.addMessage(msg.role, msg.content, msg.id);
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
            <div class="message assistant welcome-message">
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
        this.updateHistoryList();
        this.showToast('New conversation started!');
    }

    clearHistory() {
        if (confirm('Clear all chat history? This cannot be undone.')) {
            localStorage.removeItem('chatHistory');
            this.newChat();
            this.updateHistoryList();
            this.showToast('History cleared!');
        }
    }

    exportChat() {
        const messages = [];
        document.querySelectorAll('#messages .message').forEach(msg => {
            const role = msg.classList.contains('user') ? 'User' : 'AI Assistant';
            const content = msg.querySelector('.message-text').innerText;
            const time = msg.querySelector('.message-time')?.innerText || '';
            messages.push(`[${role}] ${time}\n${content}\n${'-'.repeat(50)}\n`);
        });

        if (messages.length === 0) {
            this.showToast('No messages to export');
            return;
        }

        const exportText = `Chronobiotics Chat Export\n${'='.repeat(50)}\n\n${messages.join('\n')}\nExported: ${new Date().toLocaleString()}`;
        const blob = new Blob([exportText], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `chat_export_${new Date().toISOString().slice(0, 19)}.txt`;
        a.click();
        URL.revokeObjectURL(url);
        this.showToast('Chat exported!');
    }

    showToast(message, bg = '#333') {
        const toast = document.createElement('div');
        toast.textContent = message;
        toast.style.cssText = `
            position: fixed;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            background: ${bg};
            color: white;
            padding: 8px 16px;
            border-radius: 8px;
            z-index: 10000;
            font-size: 13px;
            animation: fadeOut 2s ease forwards;
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

document.addEventListener('DOMContentLoaded', () => {
    window.agentChat = new AgentChat();
    // Initialize after a short delay to ensure all DOM elements are ready
    setTimeout(() => {
        window.agentChat.init();
    }, 100);
});