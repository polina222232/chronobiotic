/**
 * Typing Simulator - Animated typing effect
 */

class TypingSimulator {
    constructor() {
        this.isTyping = false;
        this.currentInterval = null;
        this.speed = 30;
        this.currentElement = null;
        this.currentText = '';
        this.currentIndex = 0;
    }

    async simulate(element, text, onComplete) {
        if (this.isTyping) {
            this.stop();
        }

        this.isTyping = true;
        this.currentElement = element;
        this.currentText = text;
        this.currentIndex = 0;
        element.innerHTML = '';

        return new Promise((resolve) => {
            this.currentInterval = setInterval(() => {
                if (this.currentIndex < this.currentText.length && this.isTyping) {
                    // Add next character
                    const char = this.currentText[this.currentIndex];
                    element.innerHTML += char;
                    this.currentIndex++;

                    // Auto-scroll
                    const container = document.getElementById('messagesContainer');
                    if (container && localStorage.getItem('autoScroll') !== 'false') {
                        container.scrollTop = container.scrollHeight;
                    }
                } else {
                    this.stop();
                    if (onComplete) onComplete();
                    resolve();
                }
            }, this.speed);
        });
    }

    stop() {
        this.isTyping = false;
        if (this.currentInterval) {
            clearInterval(this.currentInterval);
            this.currentInterval = null;
        }
    }

    setSpeed(speed) {
        this.speed = Math.max(10, Math.min(100, speed));
    }

    isTypingActive() {
        return this.isTyping;
    }

    skip() {
        if (this.isTyping && this.currentElement && this.currentText) {
            this.currentElement.innerHTML = this.currentText;
            this.stop();
        }
    }
}

document.addEventListener('DOMContentLoaded', () => {
    window.typingSimulator = new TypingSimulator();
});