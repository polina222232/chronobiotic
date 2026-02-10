/**
 * Chat Streaming Module
 * Handles streaming responses from the API
 */

class ChatStreaming {
    constructor(messageHandler) {
        this.messageHandler = messageHandler;
        this.isStreaming = false;
        this.currentStreamMessage = null;
        this.abortController = null;
    }

    async streamResponse(message, temperature = 0.7, maxTokens = 2048) {
        if (this.isStreaming) {
            this.stopStreaming();
        }

        this.isStreaming = true;
        this.showLoadingIndicator();

        // Create placeholder for streaming message
        const placeholderId = this.createStreamingPlaceholder();

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
                    temperature: temperature,
                    max_tokens: maxTokens
                }),
                signal: this.abortController.signal
            });

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let fullResponse = '';
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
                                fullResponse += data.chunk;
                                this.updateStreamingMessage(placeholderId, fullResponse);
                            }
                            if (data.done) {
                                if (data.citations) {
                                    this.messageHandler.updateMessage(placeholderId, fullResponse, data.citations);
                                    if (window.citationManager) {
                                        window.citationManager.show(data.citations);
                                    }
                                } else {
                                    this.messageHandler.updateMessage(placeholderId, fullResponse);
                                }
                                this.isStreaming = false;
                                this.hideLoadingIndicator();
                                return fullResponse;
                            }
                        } catch (e) {
                            console.error('Parse error:', e);
                        }
                    }
                }
            }

            this.messageHandler.updateMessage(placeholderId, fullResponse);
            this.isStreaming = false;
            this.hideLoadingIndicator();
            return fullResponse;

        } catch (error) {
            if (error.name !== 'AbortError') {
                console.error('Streaming error:', error);
                this.messageHandler.updateMessage(placeholderId, 'Sorry, I encountered an error. Please try again.');
            }
            this.isStreaming = false;
            this.hideLoadingIndicator();
            return null;
        }
    }

    createStreamingPlaceholder() {
        const messageDiv = document.createElement('div');
        const messageId = 'stream_' + Date.now();
        messageDiv.className = 'message-bubble assistant streaming';
        messageDiv.setAttribute('data-message-id', messageId);

        messageDiv.innerHTML = `
            <div class="message-avatar"><i class="fas fa-robot"></i></div>
            <div class="message-content">
                <div class="message-text">
                    <div class="typing-indicator">
                        <span></span><span></span><span></span>
                    </div>
                </div>
                <div class="message-time">Streaming...</div>
            </div>
        `;

        this.messageHandler.messageContainer.appendChild(messageDiv);
        this.messageHandler.scrollToBottom();

        // Add to messages array
        this.messageHandler.messages.push({
            id: messageId,
            role: 'assistant',
            content: '',
            timestamp: new Date().toISOString()
        });

        return messageId;
    }

    updateStreamingMessage(messageId, content) {
        const messageDiv = this.messageHandler.messageContainer.querySelector(`[data-message-id="${messageId}"]`);
        if (messageDiv) {
            const textDiv = messageDiv.querySelector('.message-text');
            if (textDiv) {
                textDiv.innerHTML = markdownRenderer.render(content);

                // Highlight code blocks
                if (window.hljs) {
                    textDiv.querySelectorAll('pre code').forEach((block) => {
                        hljs.highlightElement(block);
                    });
                }
            }

            // Update in messages array
            const message = this.messageHandler.messages.find(m => m.id === messageId);
            if (message) {
                message.content = content;
            }

            this.messageHandler.scrollToBottom();
        }
    }

    stopStreaming() {
        if (this.abortController) {
            this.abortController.abort();
        }
        this.isStreaming = false;
        this.hideLoadingIndicator();
    }

    showLoadingIndicator() {
        const loader = document.getElementById('loadingIndicator');
        if (loader) loader.style.display = 'flex';

        const sendBtn = document.getElementById('sendBtn');
        if (sendBtn) sendBtn.disabled = true;
    }

    hideLoadingIndicator() {
        const loader = document.getElementById('loadingIndicator');
        if (loader) loader.style.display = 'none';

        const sendBtn = document.getElementById('sendBtn');
        if (sendBtn) sendBtn.disabled = false;
    }

    getCookie(name) {
        let cookieValue = null;
        if (document.cookie && document.cookie !== '') {
            const cookies = document.cookie.split(';');
            for (let i = 0; i < cookies.length; i++) {
                const cookie = cookies[i].trim();
                if (cookie.substring(0, name.length + 1) === (name + '=')) {
                    cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                    break;
                }
            }
        }
        return cookieValue;
    }
}

// Initialize streaming
const chatStreaming = new ChatStreaming(messageHandler);