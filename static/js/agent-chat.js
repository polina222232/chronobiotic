/**
 * Agent Chat Main Application
 * Initializes and manages the chat interface
 */

class AgentChat {
    constructor() {
        this.messageInput = document.getElementById('messageInput');
        this.sendBtn = document.getElementById('sendBtn');
        this.newChatBtn = document.getElementById('newChatBtn');
        this.clearHistoryBtn = document.getElementById('clearHistoryBtn');
        this.exportChatBtn = document.getElementById('exportChatBtn');
        this.temperatureSlider = document.getElementById('temperatureSlider');
        this.tempValue = document.getElementById('tempValue');
        this.maxTokensInput = document.getElementById('maxTokens');
        this.streamCheckbox = document.getElementById('streamResponse');
        this.systemPrompt = document.getElementById('systemPrompt');

        this.init();
    }

    init() {
        this.setupEventListeners();
        this.loadSettings();
        this.setupSuggestionListeners();
        this.setupRegenerateListener();
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
                this.messageInput.style.height = Math.min(this.messageInput.scrollHeight, 120) + 'px';
            });
        }

        if (this.newChatBtn) {
            this.newChatBtn.addEventListener('click', () => this.newConversation());
        }

        if (this.clearHistoryBtn) {
            this.clearHistoryBtn.addEventListener('click', () => this.clearAllHistory());
        }

        if (this.exportChatBtn) {
            this.exportChatBtn.addEventListener('click', () => this.exportCurrentChat());
        }

        if (this.temperatureSlider && this.tempValue) {
            this.temperatureSlider.addEventListener('input', (e) => {
                this.tempValue.textContent = e.target.value;
                this.saveSettings();
            });
        }

        if (this.maxTokensInput) {
            this.maxTokensInput.addEventListener('change', () => this.saveSettings());
        }

        if (this.streamCheckbox) {
            this.streamCheckbox.addEventListener('change', () => this.saveSettings());
        }

        if (this.systemPrompt) {
            this.systemPrompt.addEventListener('change', () => this.saveSettings());
        }

        // Mobile sidebar toggle
        const toggleBtn = document.getElementById('mobileSidebarToggle');
        const sidebar = document.getElementById('chatSidebar');
        if (toggleBtn && sidebar) {
            toggleBtn.addEventListener('click', () => {
                sidebar.classList.toggle('show');
            });
        }

        // Settings toggle
        const settingsToggle = document.getElementById('settingsToggleBtn');
        const settingsContent = document.getElementById('settingsContent');
        if (settingsToggle && settingsContent) {
            settingsToggle.addEventListener('click', () => {
                const isVisible = settingsContent.style.display === 'block';
                settingsContent.style.display = isVisible ? 'none' : 'block';
            });
        }

        // Voice settings toggle
        const voiceSettingsToggle = document.getElementById('voiceSettingsToggle');
        const voiceSettingsContent = document.getElementById('voiceSettingsContent');
        if (voiceSettingsToggle && voiceSettingsContent) {
            voiceSettingsToggle.addEventListener('click', () => {
                const isVisible = voiceSettingsContent.style.display === 'block';
                voiceSettingsContent.style.display = isVisible ? 'none' : 'block';
            });
        }

        // Language selector
        const languageToggle = document.getElementById('languageToggleBtn');
        const languageDropdown = document.getElementById('languageDropdown');
        if (languageToggle && languageDropdown) {
            languageToggle.addEventListener('click', () => {
                languageDropdown.style.display = languageDropdown.style.display === 'block' ? 'none' : 'block';
            });

            document.querySelectorAll('.language-list li').forEach(item => {
                item.addEventListener('click', () => {
                    const lang = item.dataset.lang;
                    this.changeLanguage(lang);
                    languageDropdown.style.display = 'none';
                });
            });
        }

        // Close dropdowns when clicking outside
        document.addEventListener('click', (e) => {
            if (languageDropdown && !languageToggle.contains(e.target)) {
                languageDropdown.style.display = 'none';
            }
        });
    }

    async sendMessage() {
        const message = this.messageInput.value.trim();
        if (!message) return;

        // Add user message
        messageHandler.addMessage('user', message);

        // Clear input
        this.messageInput.value = '';
        this.messageInput.style.height = 'auto';

        // Get settings
        const temperature = this.getTemperature();
        const useStream = this.streamCheckbox ? this.streamCheckbox.checked : true;

        if (useStream) {
            // Use streaming response
            await chatStreaming.streamResponse(message, temperature);
        } else {
            // Use regular API
            await this.sendRegularRequest(message, temperature);
        }
    }

    async sendRegularRequest(message, temperature) {
        this.showLoading();

        try {
            const response = await fetch('/api/chat/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': this.getCookie('csrftoken')
                },
                body: JSON.stringify({
                    message: message,
                    temperature: temperature,
                    max_tokens: this.getMaxTokens()
                })
            });

            const data = await response.json();

            if (data.success) {
                messageHandler.addMessage('assistant', data.response, data.citations);

                if (data.citations && data.citations.length > 0 && citationManager) {
                    citationManager.show(data.citations);
                }
            } else {
                messageHandler.addMessage('assistant', 'Sorry, I encountered an error. Please try again.');
            }
        } catch (error) {
            console.error('Error:', error);
            messageHandler.addMessage('assistant', 'Network error. Please check your connection.');
        } finally {
            this.hideLoading();
        }
    }

    newConversation() {
        messageHandler.newConversation();
        this.showToast('New conversation started!');
    }

    clearAllHistory() {
        if (confirm('Are you sure you want to clear all conversation history? This cannot be undone.')) {
            localStorage.removeItem('conversations');
            localStorage.removeItem('currentConversationId');
            messageHandler.conversations = {};
            messageHandler.newConversation();
            this.showToast('All history cleared!');
        }
    }

    exportCurrentChat() {
        const format = prompt('Export format? (txt, json, html)', 'txt');
        if (format && ['txt', 'json', 'html'].includes(format)) {
            messageHandler.exportConversation(format);
        }
    }

    getTemperature() {
        return this.temperatureSlider ? parseFloat(this.temperatureSlider.value) : 0.7;
    }

    getMaxTokens() {
        return this.maxTokensInput ? parseInt(this.maxTokensInput.value) : 2048;
    }

    saveSettings() {
        const settings = {
            temperature: this.getTemperature(),
            maxTokens: this.getMaxTokens(),
            streamResponse: this.streamCheckbox ? this.streamCheckbox.checked : true,
            systemPrompt: this.systemPrompt ? this.systemPrompt.value : ''
        };
        localStorage.setItem('agentSettings', JSON.stringify(settings));
    }

    loadSettings() {
        const saved = localStorage.getItem('agentSettings');
        if (saved) {
            try {
                const settings = JSON.parse(saved);
                if (this.temperatureSlider && this.tempValue) {
                    this.temperatureSlider.value = settings.temperature;
                    this.tempValue.textContent = settings.temperature;
                }
                if (this.maxTokensInput) this.maxTokensInput.value = settings.maxTokens;
                if (this.streamCheckbox) this.streamCheckbox.checked = settings.streamResponse;
                if (this.systemPrompt) this.systemPrompt.value = settings.systemPrompt;
            } catch (e) {
                console.error('Error loading settings:', e);
            }
        }
    }

    changeLanguage(lang) {
        const langNames = {
            en: 'English', ru: 'Русский', es: 'Español',
            fr: 'Français', de: 'Deutsch'
        };

        const currentLangSpan = document.getElementById('currentLanguage');
        if (currentLangSpan) {
            currentLangSpan.textContent = langNames[lang] || lang;
        }

        // Set RTL for Arabic
        if (lang === 'ar') {
            document.body.setAttribute('dir', 'rtl');
        } else {
            document.body.setAttribute('dir', 'ltr');
        }

        localStorage.setItem('language', lang);
        this.showToast(`Language changed to ${langNames[lang]}`);
    }

    setupSuggestionListeners() {
        document.addEventListener('suggestionClick', (e) => {
            if (this.messageInput) {
                this.messageInput.value = e.detail.text;
                this.sendMessage();
            }
        });

        // Also add direct listeners to existing suggestion chips
        document.querySelectorAll('.suggestion-chip').forEach(chip => {
            chip.addEventListener('click', () => {
                if (this.messageInput) {
                    this.messageInput.value = chip.textContent;
                    this.sendMessage();
                }
            });
        });
    }

    setupRegenerateListener() {
        document.addEventListener('regenerateMessage', (e) => {
            const originalContent = e.detail.content;
            if (originalContent) {
                this.sendMessage();
            }
        });
    }

    showLoading() {
        const loader = document.getElementById('loadingIndicator');
        if (loader) loader.style.display = 'flex';
        if (this.sendBtn) this.sendBtn.disabled = true;
    }

    hideLoading() {
        const loader = document.getElementById('loadingIndicator');
        if (loader) loader.style.display = 'none';
        if (this.sendBtn) this.sendBtn.disabled = false;
    }

    showToast(message) {
        const toast = document.createElement('div');
        toast.className = 'agent-toast';
        toast.innerHTML = `<i class="fas fa-info-circle"></i> ${message}`;
        toast.style.cssText = `
            position: fixed;
            bottom: 80px;
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

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.agentChat = new AgentChat();
});