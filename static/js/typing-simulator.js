/**
 * Typing Simulator Module
 * Simulates typing animation for AI responses
 */

class TypingSimulator {
    constructor() {
        this.isTyping = false;
        this.currentInterval = null;
        this.speed = 30; // ms per character
    }

    simulateTyping(element, text, onComplete) {
        this.stop();
        this.isTyping = true;

        let index = 0;
        element.innerHTML = '';

        this.currentInterval = setInterval(() => {
            if (index < text.length && this.isTyping) {
                element.innerHTML += text[index];
                index++;

                // Auto-scroll
                const container = document.getElementById('chatMessagesContainer');
                if (container) {
                    container.scrollTop = container.scrollHeight;
                }
            } else {
                this.stop();
                if (onComplete) onComplete();
            }
        }, this.speed);
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
}

const typingSimulator = new TypingSimulator();