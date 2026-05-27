/**
 * Voice Commands - Voice command recognition
 */

class VoiceCommands {
    constructor() {
        this.recognition = null;
        this.isListening = false;
        this.commands = {
            'new chat': () => this.newChat(),
            'clear chat': () => this.clearChat(),
            'clear history': () => this.clearHistory(),
            'help': () => this.showHelp(),
            'search for': (query) => this.searchDatabase(query),
            'find': (query) => this.searchDatabase(query),
            'tell me about': (query) => this.askAbout(query),
            'what is': (query) => this.askAbout(query),
            'export chat': () => this.exportChat(),
            'settings': () => this.openSettings(),
            'voice settings': () => this.openVoiceSettings(),
            'stop listening': () => this.stopListening(),
            'cancel': () => this.stopListening()
        };
        this.init();
    }

    init() {
        this.initSpeechRecognition();
    }

    initSpeechRecognition() {
        if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
            const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
            this.recognition = new SpeechRecognition();
            this.recognition.continuous = true;
            this.recognition.interimResults = true;
            this.recognition.lang = localStorage.getItem('voiceLanguage') || 'en-US';

            this.recognition.onresult = (event) => {
                let finalTranscript = '';
                for (let i = event.resultIndex; i < event.results.length; i++) {
                    if (event.results[i].isFinal) {
                        finalTranscript += event.results[i][0].transcript;
                    }
                }
                if (finalTranscript) {
                    this.processCommand(finalTranscript.toLowerCase());
                }
            };

            this.recognition.onerror = (event) => {
                console.error('Recognition error:', event.error);
                this.stopListening();
            };

            this.recognition.onend = () => {
                this.isListening = false;
                this.updateListeningStatus(false);
            };
        }
    }

    toggleListening() {
        if (this.isListening) {
            this.stopListening();
        } else {
            this.startListening();
        }
    }

    startListening() {
        if (this.recognition && !this.isListening) {
            this.recognition.lang = localStorage.getItem('voiceLanguage') || 'en-US';
            this.recognition.start();
            this.isListening = true;
            this.updateListeningStatus(true);
            this.speakFeedback('Listening for commands');
        }
    }

    stopListening() {
        if (this.recognition && this.isListening) {
            this.recognition.stop();
            this.isListening = false;
            this.updateListeningStatus(false);
            this.speakFeedback('Stopped listening');
        }
    }

    updateListeningStatus(isListening) {
        const voiceBtn = document.getElementById('voiceBtn');
        if (voiceBtn) {
            if (isListening) {
                voiceBtn.classList.add('listening');
                voiceBtn.style.background = '#10b981';
                voiceBtn.style.color = 'white';
            } else {
                voiceBtn.classList.remove('listening');
                voiceBtn.style.background = '';
                voiceBtn.style.color = '';
            }
        }
    }

    processCommand(transcript) {
        for (const [command, action] of Object.entries(this.commands)) {
            if (transcript.includes(command)) {
                let query = '';
                if (['search for', 'find', 'tell me about', 'what is'].includes(command)) {
                    query = transcript.replace(command, '').trim();
                    if (query) action(query);
                    else action();
                } else {
                    action();
                }
                this.speakFeedback(`Executing: ${command}`);
                return;
            }
        }

        const input = document.getElementById('messageInput');
        if (input && transcript) {
            input.value = transcript;
            this.speakFeedback('Text added');
        }
    }

    newChat() {
        document.getElementById('newChatBtn')?.click();
    }

    clearChat() {
        if (confirm('Clear current conversation?')) {
            document.getElementById('newChatBtn')?.click();
        }
    }

    clearHistory() {
        if (confirm('Clear all chat history?')) {
            localStorage.removeItem('chatHistory');
            document.getElementById('newChatBtn')?.click();
            this.showToast('History cleared!');
        }
    }

    showHelp() {
        const helpText = `🎤 **Voice Commands**

**Basic:**
• "New chat" - Start new conversation
• "Clear chat" - Clear current chat
• "Clear history" - Delete all history
• "Export chat" - Save conversation

**Search:**
• "Search for [query]"
• "Find [query]"
• "Tell me about [query]"
• "What is [query]"

**Settings:**
• "Settings" - Open settings
• "Voice settings" - Voice options
• "Stop listening" - Disable voice`;

        const input = document.getElementById('messageInput');
        if (input) {
            input.value = helpText;
            this.showToast('Help displayed');
        }
    }

    searchDatabase(query) {
        if (query) {
            const input = document.getElementById('messageInput');
            if (input) {
                input.value = `Search for ${query}`;
                document.getElementById('sendBtn')?.click();
            }
        }
    }

    askAbout(query) {
        if (query) {
            const input = document.getElementById('messageInput');
            if (input) {
                input.value = `Tell me about ${query}`;
                document.getElementById('sendBtn')?.click();
            }
        }
    }

    exportChat() {
        document.getElementById('exportChatBtn')?.click();
    }

    openSettings() {
        const settingsToggle = document.getElementById('settingsToggleBtn');
        const settingsPanel = document.getElementById('settingsPanel');
        if (settingsToggle) {
            settingsToggle.click();
        } else if (settingsPanel) {
            settingsPanel.style.display = settingsPanel.style.display === 'block' ? 'none' : 'block';
        }
    }

    openVoiceSettings() {
        const voiceToggle = document.getElementById('voiceSettingsToggle');
        const voicePanel = document.getElementById('voicePanel');
        if (voiceToggle) {
            voiceToggle.click();
        } else if (voicePanel) {
            voicePanel.style.display = voicePanel.style.display === 'block' ? 'none' : 'block';
        }
    }

    speakFeedback(message) {
        if (window.speechSynthesis) {
            const utterance = new SpeechSynthesisUtterance(message);
            utterance.lang = localStorage.getItem('voiceLanguage') || 'en-US';
            utterance.rate = 1.2;
            utterance.volume = 0.8;
            window.speechSynthesis.cancel();
            window.speechSynthesis.speak(utterance);
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
            background: #10b981;
            color: white;
            padding: 8px 16px;
            border-radius: 10px;
            z-index: 10001;
            font-size: 13px;
            animation: slideUp 0.3s ease, fadeOut 0.3s ease 1.7s forwards;
        `;
        document.body.appendChild(toast);
        setTimeout(() => toast.remove(), 2000);
    }
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        window.voiceCommands = new VoiceCommands();
    });
} else {
    window.voiceCommands = new VoiceCommands();
}