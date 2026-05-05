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
            'explain': (query) => this.askAbout(query),
            'show me': (query) => this.searchDatabase(query),
            'export chat': () => this.exportChat(),
            'save chat': () => this.exportChat(),
            'settings': () => this.openSettings(),
            'voice settings': () => this.openVoiceSettings(),
            'stop listening': () => this.stopListening(),
            'cancel': () => this.stopListening()
        };
        this.init();
    }

    init() {
        this.initSpeechRecognition();
        this.setupWakeWord();
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
                if (event.error === 'no-speech') {
                    // Ignore no-speech errors
                } else {
                    this.stopListening();
                }
            };

            this.recognition.onend = () => {
                this.isListening = false;
                this.updateListeningStatus(false);
            };
        } else {
            console.warn('Speech recognition not supported');
        }
    }

    setupWakeWord() {
        // Simple wake word detection: "Hey AI" or "Hello AI"
        const wakeWords = ['hey ai', 'hello ai', 'ok ai', 'wake up'];

        // This would be implemented with a separate always-listening instance
        // For now, we'll use a manual activation button
        const voiceBtn = document.getElementById('voiceBtn');
        if (voiceBtn) {
            voiceBtn.addEventListener('dblclick', () => this.toggleListening());
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
                voiceBtn.style.background = '#28a745';
                voiceBtn.style.color = 'white';
            } else {
                voiceBtn.classList.remove('listening');
                voiceBtn.style.background = '';
                voiceBtn.style.color = '';
            }
        }
    }

    processCommand(transcript) {
        console.log('Processing command:', transcript);

        for (const [command, action] of Object.entries(this.commands)) {
            if (transcript.includes(command)) {
                let query = '';
                if (command === 'search for' || command === 'find' || command === 'tell me about' || command === 'what is' || command === 'explain' || command === 'show me') {
                    query = transcript.replace(command, '').trim();
                    if (query) {
                        action(query);
                    } else {
                        action();
                    }
                } else {
                    action();
                }
                this.speakFeedback(`Command executed: ${command}`);
                return;
            }
        }

        // If no command matched, treat as regular text input
        const input = document.getElementById('messageInput');
        if (input && transcript) {
            input.value = transcript;
            this.speakFeedback('Text added to input');
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
        const helpText = `Available voice commands:

🎤 Basic Commands:
- "New chat" - Start a new conversation
- "Clear chat" - Clear current conversation
- "Clear history" - Delete all chat history
- "Export chat" - Save current conversation
- "Settings" - Open settings panel
- "Voice settings" - Open voice settings

🔍 Search Commands:
- "Search for [query]" - Search the database
- "Find [query]" - Find information
- "Tell me about [query]" - Get information about a topic
- "What is [query]" - Explain a concept
- "Show me [query]" - Display results

🎙️ Voice Commands:
- "Stop listening" - Disable voice commands
- "Cancel" - Stop voice recognition
- "Help" - Show this help message`;

        const input = document.getElementById('messageInput');
        if (input) {
            input.value = helpText;
            this.showToast('Help displayed in chat');
        }
    }

    searchDatabase(query) {
        if (query) {
            const input = document.getElementById('messageInput');
            if (input) {
                input.value = `Search database for ${query}`;
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
        const settingsBtn = document.getElementById('settingsToggleBtn');
        const settingsPanel = document.getElementById('settingsPanel');
        if (settingsBtn) {
            settingsBtn.click();
        } else if (settingsPanel) {
            settingsPanel.style.display = settingsPanel.style.display === 'none' ? 'block' : 'none';
        }
    }

    openVoiceSettings() {
        const voiceBtn = document.getElementById('voiceSettingsToggle');
        const voicePanel = document.getElementById('voicePanel');
        if (voiceBtn) {
            voiceBtn.click();
        } else if (voicePanel) {
            voicePanel.style.display = voicePanel.style.display === 'none' ? 'block' : 'none';
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
            background: #333;
            color: white;
            padding: 6px 12px;
            border-radius: 6px;
            z-index: 10000;
            font-size: 12px;
        `;
        document.body.appendChild(toast);
        setTimeout(() => toast.remove(), 2000);
    }
}

document.addEventListener('DOMContentLoaded', () => {
    window.voiceCommands = new VoiceCommands();
});