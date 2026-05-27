/**
 * Audio Effects - Sound notifications for events
 */

class AudioEffects {
    constructor() {
        this.enabled = localStorage.getItem('soundEffects') !== 'false';
        this.audioContext = null;
        this.init();
    }

    init() {
        // Создаем audio context при первом взаимодействии
        const initAudio = () => {
            if (!this.audioContext && this.enabled) {
                this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
                console.log('Audio context initialized');
            }
        };

        document.addEventListener('click', initAudio, { once: true });
        document.addEventListener('keypress', initAudio, { once: true });
    }

    playSendSound() {
        if (!this.enabled) return;
        this.playBeep(440, 0.1);
    }

    playReceiveSound() {
        if (!this.enabled) return;
        this.playBeep(880, 0.15);
    }

    playNotificationSound() {
        if (!this.enabled) return;
        this.playBeep(660, 0.2);
    }

    playErrorSound() {
        if (!this.enabled) return;
        this.playBeep(220, 0.3);
    }

    playSuccessSound() {
        if (!this.enabled) return;
        this.playBeep(1046.5, 0.2);
    }

    playBeep(frequency, duration) {
        if (!this.audioContext) {
            console.warn('Audio context not available');
            return;
        }

        try {
            const now = this.audioContext.currentTime;
            const oscillator = this.audioContext.createOscillator();
            const gainNode = this.audioContext.createGain();

            oscillator.connect(gainNode);
            gainNode.connect(this.audioContext.destination);

            oscillator.frequency.value = frequency;
            oscillator.type = 'sine';

            gainNode.gain.value = 0.2;

            oscillator.start(now);
            gainNode.gain.exponentialRampToValueAtTime(0.00001, now + duration);
            oscillator.stop(now + duration);
        } catch (e) {
            console.warn('Audio error:', e);
        }
    }

    playChord(notes, duration) {
        if (!this.enabled || !this.audioContext) return;

        try {
            const now = this.audioContext.currentTime;
            notes.forEach(frequency => {
                const oscillator = this.audioContext.createOscillator();
                const gainNode = this.audioContext.createGain();

                oscillator.connect(gainNode);
                gainNode.connect(this.audioContext.destination);

                oscillator.frequency.value = frequency;
                oscillator.type = 'sine';
                gainNode.gain.value = 0.15;

                oscillator.start(now);
                gainNode.gain.exponentialRampToValueAtTime(0.00001, now + duration);
                oscillator.stop(now + duration);
            });
        } catch (e) {
            console.warn('Audio error:', e);
        }
    }

    toggle(enabled) {
        this.enabled = enabled;
        localStorage.setItem('soundEffects', enabled);
        if (!enabled && this.audioContext) {
            this.audioContext.close();
            this.audioContext = null;
        }
    }
}

document.addEventListener('DOMContentLoaded', () => {
    window.audioEffects = new AudioEffects();
});