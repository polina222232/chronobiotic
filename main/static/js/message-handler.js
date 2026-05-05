/**
 * Message Handler - Manages message storage and retrieval
 */

class MessageHandler {
    constructor() {
        this.messages = [];
        this.conversations = {};
        this.currentConversationId = localStorage.getItem('currentConversationId') || this.generateId();
        this.loadMessages();
        this.loadConversations();
    }

    generateId() {
        return Date.now().toString();
    }

    addMessage(role, content, metadata = {}) {
        const message = {
            id: this.generateId(),
            role: role,
            content: content,
            timestamp: new Date().toISOString(),
            metadata: metadata
        };

        if (!this.conversations[this.currentConversationId]) {
            this.conversations[this.currentConversationId] = {
                id: this.currentConversationId,
                title: content.substring(0, 50) + (content.length > 50 ? '...' : ''),
                messages: [],
                created: new Date().toISOString(),
                updated: new Date().toISOString()
            };
        }

        this.conversations[this.currentConversationId].messages.push(message);
        this.conversations[this.currentConversationId].updated = new Date().toISOString();

        if (role === 'user' && this.conversations[this.currentConversationId].messages.length === 1) {
            this.conversations[this.currentConversationId].title = content.substring(0, 50) + (content.length > 50 ? '...' : '');
        }

        this.saveConversations();
        return message;
    }

    getMessages(conversationId = null) {
        const id = conversationId || this.currentConversationId;
        return this.conversations[id]?.messages || [];
    }

    getConversation(id) {
        return this.conversations[id];
    }

    getAllConversations() {
        return Object.values(this.conversations).sort((a, b) =>
            new Date(b.updated) - new Date(a.updated)
        );
    }

    switchConversation(conversationId) {
        if (this.conversations[conversationId]) {
            this.currentConversationId = conversationId;
            localStorage.setItem('currentConversationId', conversationId);
            return this.getMessages();
        }
        return [];
    }

    newConversation() {
        this.currentConversationId = this.generateId();
        localStorage.setItem('currentConversationId', this.currentConversationId);
        return this.currentConversationId;
    }

    deleteConversation(conversationId) {
        delete this.conversations[conversationId];
        this.saveConversations();

        const remaining = Object.keys(this.conversations);
        if (remaining.length > 0) {
            this.switchConversation(remaining[0]);
        } else {
            this.newConversation();
        }
    }

    clearAllConversations() {
        this.conversations = {};
        this.newConversation();
        this.saveConversations();
    }

    updateMessage(messageId, newContent, conversationId = null) {
        const convId = conversationId || this.currentConversationId;
        const conversation = this.conversations[convId];

        if (conversation) {
            const message = conversation.messages.find(m => m.id === messageId);
            if (message) {
                message.content = newContent;
                message.edited = true;
                message.editedAt = new Date().toISOString();
                conversation.updated = new Date().toISOString();
                this.saveConversations();
                return true;
            }
        }
        return false;
    }

    deleteMessage(messageId, conversationId = null) {
        const convId = conversationId || this.currentConversationId;
        const conversation = this.conversations[convId];

        if (conversation) {
            const index = conversation.messages.findIndex(m => m.id === messageId);
            if (index !== -1) {
                conversation.messages.splice(index, 1);
                conversation.updated = new Date().toISOString();
                this.saveConversations();
                return true;
            }
        }
        return false;
    }

    saveMessages() {
        localStorage.setItem('chatMessages', JSON.stringify(this.messages));
    }

    loadMessages() {
        const saved = localStorage.getItem('chatMessages');
        if (saved) {
            this.messages = JSON.parse(saved);
        }
    }

    saveConversations() {
        localStorage.setItem('chatConversations', JSON.stringify(this.conversations));
    }

    loadConversations() {
        const saved = localStorage.getItem('chatConversations');
        if (saved) {
            this.conversations = JSON.parse(saved);
        }

        if (!this.conversations[this.currentConversationId]) {
            this.conversations[this.currentConversationId] = {
                id: this.currentConversationId,
                title: 'New conversation',
                messages: [],
                created: new Date().toISOString(),
                updated: new Date().toISOString()
            };
            this.saveConversations();
        }
    }

    exportConversation(format = 'txt', conversationId = null) {
        const convId = conversationId || this.currentConversationId;
        const conversation = this.conversations[convId];

        if (!conversation) return null;

        if (format === 'txt') {
            let text = `Chronobiotics Chat Export\n`;
            text += `${'='.repeat(50)}\n`;
            text += `Conversation: ${conversation.title}\n`;
            text += `Date: ${new Date(conversation.created).toLocaleString()}\n`;
            text += `${'='.repeat(50)}\n\n`;

            conversation.messages.forEach(msg => {
                const role = msg.role === 'user' ? 'User' : 'AI Assistant';
                const time = new Date(msg.timestamp).toLocaleTimeString();
                text += `[${role}] ${time}\n`;
                text += `${msg.content}\n`;
                text += `${'-'.repeat(30)}\n\n`;
            });

            return text;
        } else if (format === 'json') {
            return JSON.stringify(conversation, null, 2);
        } else if (format === 'html') {
            let html = `<!DOCTYPE html>
            <html>
            <head><meta charset="UTF-8"><title>Chronobiotics Chat - ${conversation.title}</title>
            <style>body{font-family:sans-serif;max-width:800px;margin:0 auto;padding:20px;background:#f5f5f5;}
            .message{margin-bottom:20px;padding:12px;border-radius:8px;}
            .user{background:#2c3e50;color:white;text-align:right;}
            .assistant{background:white;border:1px solid #ddd;}
            .time{font-size:11px;color:#999;margin-top:5px;}</style>
            </head>
            <body>
            <h1>Chronobiotics Chat Export</h1>
            <p><strong>Conversation:</strong> ${conversation.title}</p>
            <p><strong>Date:</strong> ${new Date(conversation.created).toLocaleString()}</p>
            <hr>`;

            conversation.messages.forEach(msg => {
                const role = msg.role === 'user' ? 'user' : 'assistant';
                const time = new Date(msg.timestamp).toLocaleString();
                html += `<div class="message ${role}">
                    <strong>${msg.role === 'user' ? 'You' : 'AI Assistant'}</strong>
                    <div>${msg.content.replace(/\n/g, '<br>')}</div>
                    <div class="time">${time}</div>
                </div>`;
            });

            html += `</body></html>`;
            return html;
        }

        return null;
    }

    getLastMessage() {
        const messages = this.getMessages();
        return messages[messages.length - 1];
    }

    clear() {
        this.messages = [];
        this.saveMessages();
    }
}

document.addEventListener('DOMContentLoaded', () => {
    window.messageHandler = new MessageHandler();
});