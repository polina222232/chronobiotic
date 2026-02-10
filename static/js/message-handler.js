/**
 * Message Handler Module
 * Manages chat messages, storage, and rendering
 */

class MessageHandler {
    constructor() {
        this.messages = [];
        this.messageContainer = document.getElementById('chatMessages');
        this.conversations = this.loadConversations();
        this.currentConversationId = localStorage.getItem('currentConversationId') || this.generateId();

        this.loadConversation(this.currentConversationId);
    }

    generateId() {
        return Date.now().toString();
    }

    addMessage(role, content, citations = [], metadata = {}) {
        const message = {
            id: this.generateId(),
            role: role,
            content: content,
            citations: citations,
            timestamp: new Date().toISOString(),
            metadata: metadata
        };

        this.messages.push(message);
        this.renderMessage(message);
        this.scrollToBottom();
        this.saveCurrentConversation();

        return message;
    }

    renderMessage(message) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message-bubble ${message.role}`;
        messageDiv.setAttribute('data-message-id', message.id);

        const avatar = message.role === 'user' ? '<i class="fas fa-user"></i>' : '<i class="fas fa-robot"></i>';
        const formattedContent = markdownRenderer.render(message.content);

        messageDiv.innerHTML = `
            <div class="message-avatar">${avatar}</div>
            <div class="message-content">
                <div class="message-text">${formattedContent}</div>
                <div class="message-time">${this.formatTime(message.timestamp)}</div>
                <div class="message-actions">
                    <button class="copy-message" title="Copy">
                        <i class="fas fa-copy"></i>
                    </button>
                    ${message.role === 'assistant' ? `
                        <button class="speak-message" title="Listen">
                            <i class="fas fa-volume-up"></i>
                        </button>
                        <button class="regenerate-message" title="Regenerate">
                            <i class="fas fa-redo"></i>
                        </button>
                    ` : ''}
                </div>
            </div>
        `;

        this.messageContainer.appendChild(messageDiv);

        // Add event listeners
        this.addMessageEventListeners(messageDiv, message);

        // Highlight code blocks
        if (window.hljs) {
            messageDiv.querySelectorAll('pre code').forEach((block) => {
                hljs.highlightElement(block);
            });
        }

        return messageDiv;
    }

    addMessageEventListeners(messageDiv, message) {
        const copyBtn = messageDiv.querySelector('.copy-message');
        if (copyBtn) {
            copyBtn.addEventListener('click', () => this.copyMessageContent(message.content));
        }

        const speakBtn = messageDiv.querySelector('.speak-message');
        if (speakBtn && window.speechSynthesis) {
            speakBtn.addEventListener('click', () => this.speakMessage(message.content));
        }

        const regenerateBtn = messageDiv.querySelector('.regenerate-message');
        if (regenerateBtn) {
            regenerateBtn.addEventListener('click', () => {
                const event = new CustomEvent('regenerateMessage', {
                    detail: { messageId: message.id, content: message.content }
                });
                document.dispatchEvent(event);
            });
        }
    }

    copyMessageContent(content) {
        navigator.clipboard.writeText(content).then(() => {
            this.showToast('Message copied to clipboard!');
        }).catch(err => {
            console.error('Copy failed:', err);
        });
    }

    speakMessage(content) {
        if (window.speechSynthesis) {
            const utterance = new SpeechSynthesisUtterance(content);

            // Get voice settings
            const voiceLang = localStorage.getItem('voiceLanguage') || 'en-US';
            const voiceSpeed = parseFloat(localStorage.getItem('voiceSpeed') || '1');

            utterance.lang = voiceLang;
            utterance.rate = voiceSpeed;

            window.speechSynthesis.cancel();
            window.speechSynthesis.speak(utterance);
        }
    }

    updateMessage(messageId, newContent, citations = null) {
        const message = this.messages.find(m => m.id === messageId);
        if (message) {
            message.content = newContent;
            if (citations) message.citations = citations;

            // Update DOM
            const messageDiv = this.messageContainer.querySelector(`[data-message-id="${messageId}"]`);
            if (messageDiv) {
                const textDiv = messageDiv.querySelector('.message-text');
                if (textDiv) {
                    textDiv.innerHTML = markdownRenderer.render(newContent);

                    // Re-highlight code blocks
                    if (window.hljs) {
                        textDiv.querySelectorAll('pre code').forEach((block) => {
                            hljs.highlightElement(block);
                        });
                    }
                }
            }

            this.saveCurrentConversation();
        }
    }

    formatTime(timestamp) {
        const date = new Date(timestamp);
        const now = new Date();
        const diff = now - date;

        if (diff < 60000) return 'Just now';
        if (diff < 3600000) return `${Math.floor(diff / 60000)} min ago`;
        if (diff < 86400000) return `${Math.floor(diff / 3600000)} hours ago`;

        return date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    }

    scrollToBottom() {
        const container = document.getElementById('chatMessagesContainer');
        if (container) {
            container.scrollTop = container.scrollHeight;
        }
    }

    clearMessages() {
        this.messages = [];
        if (this.messageContainer) {
            this.messageContainer.innerHTML = '';
        }
        this.saveCurrentConversation();
    }

    saveCurrentConversation() {
        const conversation = {
            id: this.currentConversationId,
            title: this.getConversationTitle(),
            messages: this.messages,
            timestamp: new Date().toISOString(),
            updatedAt: new Date().toISOString()
        };

        this.conversations[this.currentConversationId] = conversation;
        localStorage.setItem('conversations', JSON.stringify(this.conversations));
        localStorage.setItem('currentConversationId', this.currentConversationId);

        // Update conversation list
        this.updateConversationList();
    }

    loadConversations() {
        const saved = localStorage.getItem('conversations');
        return saved ? JSON.parse(saved) : {};
    }

    loadConversation(conversationId) {
        if (this.conversations[conversationId]) {
            this.messages = this.conversations[conversationId].messages || [];
            this.renderAllMessages();
        } else {
            this.messages = [];
            this.renderWelcomeMessage();
        }
        this.currentConversationId = conversationId;
    }

    renderAllMessages() {
        if (this.messageContainer) {
            this.messageContainer.innerHTML = '';
            this.messages.forEach(message => {
                this.renderMessage(message);
            });
            this.scrollToBottom();
        }
    }

    renderWelcomeMessage() {
        if (this.messageContainer && this.messageContainer.children.length === 0) {
            const welcomeHtml = `
                <div class="message-bubble assistant welcome">
                    <div class="message-avatar"><i class="fas fa-robot"></i></div>
                    <div class="message-content">
                        <div class="message-text">
                            <h2>👋 Welcome to Chronobiotics AI!</h2>
                            <p>I'm your specialized assistant for the Chronobiotics Database. I can help you with:</p>
                            <div class="welcome-features">
                                <div class="feature"><i class="fas fa-flask"></i> Compound information</div>
                                <div class="feature"><i class="fas fa-dna"></i> Molecular targets & mechanisms</div>
                                <div class="feature"><i class="fas fa-file-alt"></i> Research articles & citations</div>
                                <div class="feature"><i class="fas fa-clock"></i> Circadian rhythm modulation</div>
                            </div>
                            <div class="suggestions">
                                <p><strong>Try asking:</strong></p>
                                <div class="suggestion-chips">
                                    <span class="suggestion-chip">What are chronobiotics?</span>
                                    <span class="suggestion-chip">Tell me about melatonin</span>
                                    <span class="suggestion-chip">How do KL001 and KS15 work?</span>
                                    <span class="suggestion-chip">Show me clock gene targets</span>
                                </div>
                            </div>
                        </div>
                        <div class="message-time">Online</div>
                    </div>
                </div>
            `;
            this.messageContainer.innerHTML = welcomeHtml;

            // Add suggestion chip listeners
            document.querySelectorAll('.suggestion-chip').forEach(chip => {
                chip.addEventListener('click', () => {
                    const event = new CustomEvent('suggestionClick', { detail: { text: chip.textContent } });
                    document.dispatchEvent(event);
                });
            });
        }
    }

    getConversationTitle() {
        if (this.messages.length > 0) {
            const firstUserMessage = this.messages.find(m => m.role === 'user');
            if (firstUserMessage) {
                let title = firstUserMessage.content.substring(0, 50);
                if (title.length > 50) title = title.substring(0, 47) + '...';
                return title;
            }
        }
        return 'New conversation';
    }

    updateConversationList() {
        const historyList = document.getElementById('historyList');
        if (!historyList) return;

        const conversations = Object.values(this.conversations)
            .sort((a, b) => new Date(b.updatedAt) - new Date(a.updatedAt))
            .slice(0, 20);

        historyList.innerHTML = conversations.map(conv => `
            <li class="history-item ${conv.id === this.currentConversationId ? 'active' : ''}"
                data-chat-id="${conv.id}">
                <i class="fas fa-message"></i>
                <span class="history-title">${this.escapeHtml(conv.title)}</span>
                <span class="history-date">${this.formatDateShort(conv.updatedAt)}</span>
            </li>
        `).join('');

        // Add click listeners
        historyList.querySelectorAll('.history-item').forEach(item => {
            item.addEventListener('click', () => {
                const chatId = item.dataset.chatId;
                this.switchConversation(chatId);
            });
        });
    }

    switchConversation(conversationId) {
        this.currentConversationId = conversationId;
        this.messages = this.conversations[conversationId]?.messages || [];
        this.renderAllMessages();
        localStorage.setItem('currentConversationId', conversationId);
        this.updateConversationList();
    }

    newConversation() {
        this.currentConversationId = this.generateId();
        this.messages = [];
        this.renderWelcomeMessage();
        this.saveCurrentConversation();
        this.updateConversationList();
    }

    deleteConversation(conversationId) {
        delete this.conversations[conversationId];
        localStorage.setItem('conversations', JSON.stringify(this.conversations));

        const remainingIds = Object.keys(this.conversations);
        if (remainingIds.length > 0) {
            this.switchConversation(remainingIds[0]);
        } else {
            this.newConversation();
        }
    }

    exportConversation(format = 'txt') {
        const conversation = this.conversations[this.currentConversationId];
        if (!conversation) return;

        let content = '';
        const date = new Date().toISOString().slice(0, 19).replace(/:/g, '-');

        if (format === 'txt') {
            content = `Chronobiotics Chat Export\n`;
            content += `Date: ${new Date().toLocaleString()}\n`;
            content += `Conversation: ${conversation.title}\n`;
            content += `${'='.repeat(50)}\n\n`;

            conversation.messages.forEach(msg => {
                const role = msg.role === 'user' ? 'You' : 'AI Assistant';
                content += `[${role}] ${new Date(msg.timestamp).toLocaleTimeString()}\n`;
                content += `${msg.content}\n\n`;
                content += `${'-'.repeat(30)}\n\n`;
            });

            this.downloadFile(content, `chronobiotics_chat_${date}.txt`, 'text/plain');
        } else if (format === 'json') {
            content = JSON.stringify(conversation, null, 2);
            this.downloadFile(content, `chronobiotics_chat_${date}.json`, 'application/json');
        } else if (format === 'html') {
            content = this.generateHTMLExport(conversation);
            this.downloadFile(content, `chronobiotics_chat_${date}.html`, 'text/html');
        }

        this.showToast('Conversation exported!');
    }

    generateHTMLExport(conversation) {
        let html = `<!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Chronobiotics Chat - ${conversation.title}</title>
            <style>
                body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; background: #f7f7f8; }
                .message { margin-bottom: 20px; padding: 12px; border-radius: 12px; }
                .user { background: linear-gradient(135deg, #667eea, #764ba2); color: white; text-align: right; }
                .assistant { background: white; box-shadow: 0 1px 2px rgba(0,0,0,0.1); }
                .time { font-size: 11px; color: #999; margin-top: 5px; }
                h1 { color: #333; }
                hr { margin: 20px 0; }
            </style>
        </head>
        <body>
            <h1>Chronobiotics Chat Export</h1>
            <p><strong>Conversation:</strong> ${this.escapeHtml(conversation.title)}</p>
            <p><strong>Date:</strong> ${new Date(conversation.timestamp).toLocaleString()}</p>
            <hr>`;

        conversation.messages.forEach(msg => {
            html += `
            <div class="message ${msg.role}">
                <div><strong>${msg.role === 'user' ? 'You' : 'AI Assistant'}</strong></div>
                <div>${markdownRenderer.render(msg.content)}</div>
                <div class="time">${new Date(msg.timestamp).toLocaleString()}</div>
            </div>`;
        });

        html += `</body></html>`;
        return html;
    }

    downloadFile(content, filename, mimeType) {
        const blob = new Blob([content], { type: mimeType });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        a.click();
        URL.revokeObjectURL(url);
    }

    formatDateShort(dateString) {
        const date = new Date(dateString);
        const now = new Date();
        const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
        const yesterday = new Date(today.getTime() - 86400000);

        if (date >= today) return 'Today';
        if (date >= yesterday) return 'Yesterday';
        return date.toLocaleDateString([], { month: 'short', day: 'numeric' });
    }

    showToast(message) {
        const toast = document.createElement('div');
        toast.className = 'message-toast';
        toast.innerHTML = `<i class="fas fa-check-circle"></i> ${message}`;
        toast.style.cssText = `
            position: fixed;
            bottom: 100px;
            right: 20px;
            background: #333;
            color: white;
            padding: 10px 20px;
            border-radius: 8px;
            z-index: 10000;
            animation: fadeOut 2s ease;
            font-size: 14px;
        `;
        document.body.appendChild(toast);
        setTimeout(() => toast.remove(), 2000);
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Initialize message handler
const messageHandler = new MessageHandler();