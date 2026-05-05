//**
 * Voice Player - Text-to-speech with gender and language selection
 */

class VoicePlayer {
    constructor() {
        this.synth = window.speechSynthesis;
        this.voices = [];
        this.currentGender = localStorage.getItem('voiceGender') || 'female';
        this.currentLanguage = localStorage.getItem('voiceLanguage') || 'en-US';
        this.currentPitch = parseFloat(localStorage.getItem('voicePitch') || '1');
        this.currentRate = parseFloat(localStorage.getItem('voiceSpeed') || '1');
        this.init();
    }

    init() {
        this.loadVoices();
        if (this.synth) {
            this.synth.onvoiceschanged = () => this.loadVoices();
        }
        this.setupControls();
    }

    loadVoices() {
        this.voices = this.synth.getVoices();
        console.log('Voices loaded:', this.voices.length);
    }

    setupControls() {
        // Gender buttons
        const genderBtns = document.querySelectorAll('.gender-btn');
        genderBtns.forEach(btn => {
            if (btn.dataset.gender === this.currentGender) {
                btn.classList.add('active');
            }
            btn.addEventListener('click', () => {
                genderBtns.forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                this.currentGender = btn.dataset.gender;
                localStorage.setItem('voiceGender', this.currentGender);
            });
        });

        // Voice language
        const voiceLang = document.getElementById('voiceLanguage');
        if (voiceLang) {
            voiceLang.value = this.currentLanguage;
            voiceLang.addEventListener('change', (e) => {
                this.currentLanguage = e.target.value;
                localStorage.setItem('voiceLanguage', this.currentLanguage);
            });
        }

        // Voice speed
        const voiceSpeed = document.getElementById('voiceSpeed');
        const voiceSpeedValue = document.getElementById('voiceSpeedValue');
        if (voiceSpeed && voiceSpeedValue) {
            voiceSpeed.value = this.currentRate;
            voiceSpeedValue.textContent = this.currentRate;
            voiceSpeed.addEventListener('input', (e) => {
                this.currentRate = parseFloat(e.target.value);
                voiceSpeedValue.textContent = this.currentRate;
                localStorage.setItem('voiceSpeed', this.currentRate);
            });
        }

        // Voice pitch
        const voicePitch = document.getElementById('voicePitch');
        const voicePitchValue = document.getElementById('voicePitchValue');
        if (voicePitch && voicePitchValue) {
            voicePitch.value = this.currentPitch;
            voicePitchValue.textContent = this.currentPitch;
            voicePitch.addEventListener('input', (e) => {
                this.currentPitch = parseFloat(e.target.value);
                voicePitchValue.textContent = this.currentPitch;
                localStorage.setItem('voicePitch', this.currentPitch);
            });
        }

        // Auto-play
        const autoPlay = document.getElementById('autoPlayVoice');
        if (autoPlay) {
            autoPlay.checked = localStorage.getItem('autoPlayVoice') !== 'false';
        }

        // Test button
        const testBtn = document.getElementById('testVoiceBtn');
        if (testBtn) {
            testBtn.addEventListener('click', () => {
                this.speak('Hello! This is a test of the voice system.');
            });
        }
    }

    speak(text, onEnd = null) {
        if (!this.synth || !text) return;

        // Cancel any ongoing speech
        this.synth.cancel();

        const utterance = new SpeechSynthesisUtterance(text);
        utterance.lang = this.currentLanguage;
        utterance.rate = this.currentRate;
        utterance.pitch = this.currentPitch;

        // Find appropriate voice
        const langPrefix = this.currentLanguage.split('-')[0];
        const availableVoices = this.voices.filter(v => v.lang.startsWith(langPrefix));

        let selectedVoice = null;
        if (this.currentGender === 'female') {
            selectedVoice = availableVoices.find(v =>
                v.name.toLowerCase().includes('female') ||
                v.name.toLowerCase().includes('samantha') ||
                v.name.toLowerCase().includes('google uk english female') ||
                v.name.toLowerCase().includes('zira')
            );
        } else {
            selectedVoice = availableVoices.find(v =>
                v.name.toLowerCase().includes('male') ||
                v.name.toLowerCase().includes('google uk english male') ||
                v.name.toLowerCase().includes('david')
            );
        }

        if (selectedVoice) {
            utterance.voice = selectedVoice;
        } else if (availableVoices.length > 0) {
            utterance.voice = availableVoices[0];
        }

        utterance.onstart = () => {
            const playBtn = document.getElementById('voicePlayBtn');
            if (playBtn) {
                playBtn.innerHTML = '⏹️';
                playBtn.classList.add('playing');
            }
        };

        utterance.onend = () => {
            const playBtn = document.getElementById('voicePlayBtn');
            if (playBtn) {
                playBtn.innerHTML = '🔊';
                playBtn.classList.remove('playing');
            }
            if (onEnd) onEnd();
        };

        utterance.onerror = (event) => {
            console.error('Speech error:', event);
            const playBtn = document.getElementById('voicePlayBtn');
            if (playBtn) {
                playBtn.innerHTML = '🔊';
                playBtn.classList.remove('playing');
            }
        };

        this.synth.speak(utterance);
    }

    stop() {
        if (this.synth) {
            this.synth.cancel();
        }
    }
}

document.addEventListener('DOMContentLoaded', () => {
    window.voicePlayer = new VoicePlayer();
});