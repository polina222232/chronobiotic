/**
 * Voice Recorder Module
 * Handles voice recording and speech-to-text
 */

class VoiceRecorder {
    constructor() {
        this.mediaRecorder = null;
        this.audioChunks = [];
        this.isRecording = false;
        this.stream = null;

        this.recordBtn = document.getElementById('voiceRecordBtn');
        this.stopBtn = document.getElementById('voiceStopBtn');
        this.statusIndicator = document.getElementById('voiceStatusIndicator');
        this.waveContainer = document.getElementById('voiceWaveContainer');

        this.init();
    }

    init() {
        if (this.recordBtn) {
            this.recordBtn.addEventListener('click', () => this.startRecording());
        }

        if (this.stopBtn) {
            this.stopBtn.addEventListener('click', () => this.stopRecording());
        }
    }

    async startRecording() {
        try {
            this.stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            this.mediaRecorder = new MediaRecorder(this.stream);
            this.audioChunks = [];

            this.mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    this.audioChunks.push(event.data);
                }
            };

            this.mediaRecorder.onstop = () => {
                this.processRecording();
            };

            this.mediaRecorder.start(100);
            this.isRecording = true;

            this.updateUI(true);
            this.startWaveAnimation();

            if (this.statusIndicator) {
                this.statusIndicator.innerHTML = '<span class="voice-status-text recording">🔴 Recording... Speak now</span>';
            }

        } catch (error) {
            console.error('Microphone error:', error);
            this.showError('Unable to access microphone. Please check permissions.');
        }
    }

    stopRecording() {
        if (this.mediaRecorder && this.isRecording) {
            this.mediaRecorder.stop();
            this.isRecording = false;

            if (this.stream) {
                this.stream.getTracks().forEach(track => track.stop());
            }

            this.updateUI(false);
            this.stopWaveAnimation();

            if (this.statusIndicator) {
                this.statusIndicator.innerHTML = '<span class="voice-status-text">🎤 Processing audio...</span>';
            }
        }
    }

    processRecording() {
        const audioBlob = new Blob(this.audioChunks, { type: 'audio/wav' });
        this.transcribeAudio(audioBlob);
    }

    async transcribeAudio(audioBlob) {
        // Simulate speech recognition
        const mockTranscripts = [
            "What are the main classes of chronobiotics?",
            "Tell me about melatonin and its effects.",
            "How do KL001 and KS15 work?",
            "Show me the molecular targets of circadian rhythm modulators.",
            "What are the FDA-approved chronobiotics?",
            "Explain the mechanism of ramelteon.",
            "Show me research articles about chronobiotics.",
            "What are the clinical applications of melatonin?"
        ];

        const transcript = mockTranscripts[Math.floor(Math.random() * mockTranscripts.length)];

        // Simulate processing delay
        await this.delay(1000);

        const messageInput = document.getElementById('messageInput');
        if (messageInput) {
            messageInput.value = transcript;
            messageInput.dispatchEvent(new Event('input'));

            if (this.statusIndicator) {
                this.statusIndicator.innerHTML = '<span class="voice-status-text">🎤 Ready</span>';
            }

            // Auto-send after short delay
            setTimeout(() => {
                const sendBtn = document.getElementById('sendBtn');
                if (sendBtn) sendBtn.click();
            }, 500);
        }

        console.log('Transcribed:', transcript);
    }

    startWaveAnimation() {
        if (this.waveContainer) {
            this.waveContainer.style.display = 'flex';

            // Animate waves
            const waves = this.waveContainer.querySelectorAll('.voice-wave');
            waves.forEach((wave, index) => {
                wave.style.animationDelay = `${index * 0.1}s`;
            });
        }
    }

    stopWaveAnimation() {
        if (this.waveContainer) {
            this.waveContainer.style.display = 'none';
        }
    }

    updateUI(isRecording) {
        if (this.recordBtn) {
            this.recordBtn.style.display = isRecording ? 'none' : 'flex';
        }

        if (this.stopBtn) {
            this.stopBtn.style.display = isRecording ? 'flex' : 'none';
        }

        if (isRecording) {
            this.recordBtn.classList.add('recording');
        } else {
            this.recordBtn.classList.remove('recording');
        }
    }

    showError(message) {
        if (this.statusIndicator) {
            this.statusIndicator.innerHTML = `<span class="voice-status-text error">⚠️ ${message}</span>`;
            setTimeout(() => {
                if (this.statusIndicator) {
                    this.statusIndicator.innerHTML = '<span class="voice-status-text">🎤 Click to speak</span>';
                }
            }, 3000);
        }
    }

    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}

const voiceRecorder = new VoiceRecorder();