/**
 * Voice Commands Module
 * Handles voice command recognition
 */

class VoiceCommands {
    constructor() {
        this.commands = {
            'new chat': () => this.newChat(),
            'clear chat': () => this.clearChat(),
            'help': () => this.showHelp(),
            'search for': (query) => this.searchDatabase(query),
            'tell me about': (query) => this.askAbout(query),
            'what is': (query) => this.askAbout(query),
            'show me': (query) => this.searchDatabase(query),
            'export chat': () => this.exportChat(),
            'settings': () => this.openSettings(),
            'voice settings': () => this.openVoiceSettings()
        };
    }

    parseCommand(transcript) {
        const lowerTranscript = transcript.toLowerCase();

        for (const [command, action] of Object.entries(this.commands)) {
            if (lowerTranscript.startsWith(command)) {
                const query = lowerTranscript.substring(command.length).trim();
                return { action, query };
            }

            if (lowerTranscript.includes(command)) {
                const query = lowerTranscript.replace(command, '').trim();
                return { action, query };
            }
        }

        return null;
    }

    execute(transcript) {
        const command = this.parseCommand(transcript);

        if (command) {
            if (command.query) {
                command.action(command.query);
            } else {
                command.action();
            }
            return true;
        }

        return false;
    }

    newChat() {
        const newChatBtn = document.getElementById('newChatBtn');
        if (newChatBtn) newChatBtn.click();
        this.speakFeedback('Starting new conversation');
    }

    clearChat() {
        if (confirm('Clear current conversation?')) {
            const newChatBtn = document.getElementById('newChatBtn');
            if (newChatBtn) newChatBtn.click();
            this.speakFeedback('Chat cleared');
        }
    }

    showHelp() {
        const helpText = `Available voice commands:
        - New chat
        - Clear chat
        - Search for [query]
        - Tell me about [query]
        - Show me [query]
        - Export chat
        - Settings
        - Voice settings`;

        const messageInput = document.getElementById('messageInput');
        if (messageInput) {
            messageInput.value = helpText;
        }

        this.speakFeedback('Help displayed in chat');
    }

    searchDatabase(query) {
        if (query) {
            const messageInput = document.getElementById('messageInput');
            if (messageInput) {
                messageInput.value = `Search database for ${query}`;
                const sendBtn = document.getElementById('sendBtn');
                if (sendBtn) sendBtn.click();
            }
            this.speakFeedback(`Searching for ${query}`);
        }
    }

    askAbout(query) {
        if (query) {
            const messageInput = document.getElementById('messageInput');
            if (messageInput) {
                messageInput.value = `Tell me about ${query}`;
                const sendBtn = document.getElementById('sendBtn');
                if (sendBtn) sendBtn.click();
            }
            this.speakFeedback(`Asking about ${query}`);
        }
    }

    exportChat() {
        const exportBtn = document.getElementById('exportChatBtn');
        if (exportBtn) exportBtn.click();
        this.speakFeedback('Exporting chat');
    }

    openSettings() {
        const settingsToggle = document.getElementById('settingsToggleBtn');
        if (settingsToggle) settingsToggle.click();
        this.speakFeedback('Settings opened');
    }

    openVoiceSettings() {
        const voiceSettingsToggle = document.getElementById('voiceSettingsToggle');
        if (voiceSettingsToggle) voiceSettingsToggle.click();
        this.speakFeedback('Voice settings opened');
    }

    speakFeedback(message) {
        if (window.speechSynthesis) {
            const utterance = new SpeechSynthesisUtterance(message);
            utterance.lang = 'en-US';
            utterance.rate = 1.2;
            window.speechSynthesis.cancel();
            window.speechSynthesis.speak(utterance);
        }
    }
}

const voiceCommands = new VoiceCommands();