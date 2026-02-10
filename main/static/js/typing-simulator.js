class TypingSimulator {
    async simulate(element, text, onComplete) {
        element.innerHTML = '';
        for (let i = 0; i < text.length; i++) {
            element.innerHTML += text[i];
            await this.delay(30);
        }
        if (onComplete) onComplete();
    }

    delay(ms) {
        return new Promise(r => setTimeout(r, ms));
    }
}

const typingSimulator = new TypingSimulator();