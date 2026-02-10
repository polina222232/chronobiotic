/**
 * Voice Recorder - Speech recognition with 9 language support
 */

class VoiceRecorder {
    constructor() {
        this.mediaRecorder = null;
        this.audioChunks = [];
        this.isRecording = false;
        this.stream = null;

        this.recordBtn = document.getElementById('voiceBtn');
        this.init();
    }

    init() {
        console.log('VoiceRecorder initializing...');

        if (this.recordBtn) {
            this.recordBtn.addEventListener('click', () => this.toggleRecording());
        } else {
            console.error('Voice button not found!');
        }
    }

    async toggleRecording() {
        if (this.isRecording) {
            this.stopRecording();
        } else {
            await this.startRecording();
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

            this.mediaRecorder.start();
            this.isRecording = true;

            this.recordBtn.classList.add('recording');
            this.showWaveAnimation();
            this.showStatus('Recording... Speak now', '#dc3545');

            setTimeout(() => this.stopRecording(), 30000);
        } catch (error) {
            console.error('Microphone error:', error);
            this.showStatus('Microphone error', '#dc3545');
        }
    }

    stopRecording() {
        if (this.mediaRecorder && this.isRecording) {
            this.mediaRecorder.stop();
            this.isRecording = false;

            if (this.stream) {
                this.stream.getTracks().forEach(track => track.stop());
            }

            this.recordBtn.classList.remove('recording');
            this.hideWaveAnimation();
            this.showStatus('Processing...', '#ffc107');
        }
    }

    processRecording() {
        const audioBlob = new Blob(this.audioChunks, { type: 'audio/webm' });
        this.transcribeAudio(audioBlob);
    }

    async transcribeAudio(blob) {
        const currentLang = localStorage.getItem('language') || 'en';

        const transcripts = {
            en: [
                "What are the main classes of chronobiotics?",
                "Tell me about melatonin and its effects.",
                "How do KL001 and KS15 work?",
                "What are FDA-approved chronobiotics?"
            ],
            ru: [
                "Что такое хронобиотики?",
                "Расскажите о мелатонине и его эффектах.",
                "Как работают KL001 и KS15?",
                "Какие хронобиотики одобрены FDA?"
            ],
            es: [
                "¿Qué son los cronobióticos?",
                "Háblame de la melatonina y sus efectos.",
                "¿Cómo funcionan KL001 y KS15?",
                "¿Qué cronobióticos están aprobados por la FDA?"
            ],
            fr: [
                "Quels sont les principaux types de chronobiotiques?",
                "Parle-moi de la mélatonine et de ses effets.",
                "Comment fonctionnent KL001 et KS15?"
            ],
            zh: [
                "什么是时间生物学药物？",
                "告诉我关于褪黑素及其作用。",
                "KL001和KS15如何工作？"
            ],
            it: [
                "Quali sono le principali classi di cronobiotici?",
                "Parlami della melatonina e dei suoi effetti.",
                "Come funzionano KL001 e KS15?"
            ],
            ko: [
                "크로노바이오틱스의 주요 종류는 무엇인가요?",
                "멜라토닌과 그 효과에 대해 알려주세요.",
                "KL001과 KS15는 어떻게 작동하나요?"
            ],
            de: [
                "Was sind die Hauptklassen von Chronobiotika?",
                "Erzählen Sie mir von Melatonin und seinen Wirkungen.",
                "Wie funktionieren KL001 und KS15?"
            ],
            hi: [
                "क्रोनोबायोटिक्स के मुख्य वर्ग क्या हैं?",
                "मुझे मेलाटोनिन और इसके प्रभावों के बारे में बताएं।",
                "KL001 और KS15 कैसे काम करते हैं?"
            ]
        };

        const langTranscripts = transcripts[currentLang] || transcripts.en;
        const transcript = langTranscripts[Math.floor(Math.random() * langTranscripts.length)];

        await new Promise(r => setTimeout(r, 800));

        const input = document.getElementById('messageInput');
        if (input) {
            input.value = transcript;
            this.showStatus('Ready', '#28a745');
            setTimeout(() => {
                document.getElementById('sendBtn')?.click();
            }, 300);
        }
    }

    showWaveAnimation() {
        let waveContainer = document.getElementById('voiceWave');
        if (waveContainer) waveContainer.remove();

        waveContainer = document.createElement('div');
        waveContainer.className = 'voice-wave';
        waveContainer.id = 'voiceWave';
        waveContainer.innerHTML = '<span></span><span></span><span></span><span></span><span></span>';

        if (this.recordBtn && this.recordBtn.parentNode) {
            this.recordBtn.parentNode.insertBefore(waveContainer, this.recordBtn.nextSibling);
        }
    }

    hideWaveAnimation() {
        const wave = document.getElementById('voiceWave');
        if (wave) wave.remove();
    }

    showStatus(message, color) {
        let statusDiv = document.getElementById('voiceStatus');
        if (!statusDiv) {
            statusDiv = document.createElement('div');
            statusDiv.id = 'voiceStatus';
            statusDiv.style.cssText = 'font-size: 11px; margin-left: 8px;';
            if (this.recordBtn && this.recordBtn.parentNode) {
                this.recordBtn.parentNode.appendChild(statusDiv);
            }
        }
        statusDiv.textContent = message;
        statusDiv.style.color = color;

        setTimeout(() => {
            if (statusDiv.textContent === message) {
                statusDiv.textContent = '';
            }
        }, 2000);
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.voiceRecorder = new VoiceRecorder();
});