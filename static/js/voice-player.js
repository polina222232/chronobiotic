/**
 * Voice Player Module
 * Handles text-to-speech playback
 */

class VoicePlayer {
    constructor() {
        this.synth = window.speechSynthesis;
        this.isPlaying = false;
        this.currentUtterance = null;
        this.voiceSettings = this.loadSettings();

        this.init();
    }

    init() {
        this.loadVoices();

        if (this.synth) {
            this.synth.onvoiceschanged = () => this.loadVoices();
        }

        // Setup voice settings controls
        this.setupSettingsControls();
    }

    loadVoices() {
        this.voices = this.synth.getVoices();
    }

    setupSettingsControls() {
        const voiceLang = document.getElementById('voiceLanguage');
        const voiceType = document.getElementById('voiceType');
        const voiceSpeed = document.getElementById('voiceSpeed');
        const autoPlay = document.getElementById('autoPlayVoice');

        if (voiceLang) {
            voiceLang.value = this.voiceSettings.language;
            voiceLang.addEventListener('change', (e) => {
                this.voiceSettings.language = e.target.value;
                this.saveSettings();
            });
        }

        if (voiceType) {
            voiceType.value = this.voiceSettings.type;
            voiceType.addEventListener('change', (e) => {
                this.voiceSettings.type = e.target.value;
                this.saveSettings();
            });
        }

        if (voiceSpeed) {
            voiceSpeed.value = this.voiceSettings.speed;
            voiceSpeed.addEventListener('input', (e) => {
                this.voiceSettings.speed = parseFloat(e.target.value);
                this.saveSettings();
            });
        }

        if (autoPlay) {
            autoPlay.checked = this.voiceSettings.autoPlay;
            autoPlay.addEventListener('change', (e) => {
                this.voiceSettings.autoPlay = e.target.checked;
                this.saveSettings();
            });
        }
    }

    loadSettings() {
        const saved = localStorage.getItem('voiceSettings');
        if (saved) {
            try {
                return JSON.parse(saved);
            } catch (e) {
                console.error('Error loading voice settings:', e);
            }
        }
        return {
            language: 'en-US',
            type: 'female',
            speed: 1,
            autoPlay: true
        };
    }

    saveSettings() {
        localStorage.setItem('voiceSettings', JSON.stringify(this.voiceSettings));
    }

    speak(text, onEnd = null) {
        if (!this.synth) {
            console.warn('Speech synthesis not supported');
            return;
        }

        this.stop();

        this.currentUtterance = new SpeechSynthesisUtterance(text);
        this.currentUtterance.lang = this.voiceSettings.language;
        this.currentUtterance.rate = this.voiceSettings.speed;

        // Select voice based on settings
        const selectedVoice = this.selectVoice();
        if (selectedVoice) {
            this.currentUtterance.voice = selectedVoice;
        }

        this.currentUtterance.onstart = () => {
            this.isPlaying = true;
            this.updatePlayButton(true);
        };

        this.currentUtterance.onend = () => {
            this.isPlaying = false;
            this.updatePlayButton(false);
            if (onEnd) onEnd();
        };

        this.currentUtterance.onerror = (event) => {
            console.error('Speech error:', event);
            this.isPlaying = false;
            this.updatePlayButton(false);
        };

        this.synth.speak(this.currentUtterance);
    }

    selectVoice() {
        const availableVoices = this.voices.filter(v => v.lang.startsWith(this.voiceSettings.language.split('-')[0]));

        if (this.voiceSettings.type === 'female') {
            const femaleVoice = availableVoices.find(v =>
                v.name.toLowerCase().includes('female') ||
                v.name.toLowerCase().includes('samantha') ||
                v.name.toLowerCase().includes('google uk english female')
            );
            if (femaleVoice) return femaleVoice;
        } else {
            const maleVoice = availableVoices.find(v =>
                v.name.toLowerCase().includes('male') ||
                v.name.toLowerCase().includes('google uk english male')
            );
            if (maleVoice) return maleVoice;
        }

        return availableVoices[0] || this.voices[0];
    }

    stop() {
        if (this.synth && this.isPlaying) {
            this.synth.cancel();
            this.isPlaying = false;
            this.updatePlayButton(false);
        }
    }

    updatePlayButton(isPlaying) {
        const playBtn = document.getElementById('voicePlayBtn');
        if (playBtn) {
            if (isPlaying) {
                playBtn.innerHTML = '<i class="fas fa-stop"></i>';
                playBtn.title = 'Stop';
                playBtn.classList.add('playing');
            } else {
                playBtn.innerHTML = '<i class="fas fa-play"></i>';
                playBtn.title = 'Play last response';
                playBtn.classList.remove('playing');
            }
        }
    }

    playLastResponse(responseText) {
        if (this.voiceSettings.autoPlay && responseText) {
            this.speak(responseText);
        }
    }
}

const voicePlayer = new VoicePlayer();