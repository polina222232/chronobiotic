

class LLMManager {
    constructor() {
        this.currentModel = localStorage.getItem('selectedModel') || 'bloom';
        this.init();
    }

    init() {
        const modelSelect = document.getElementById('modelSelect');
        if (modelSelect) {
            modelSelect.value = this.currentModel;
            modelSelect.addEventListener('change', (e) => {
                this.currentModel = e.target.value;
                localStorage.setItem('selectedModel', this.currentModel);
                this.showToast(`Switched to ${modelSelect.options[modelSelect.selectedIndex].text}`);
            });
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

    getCurrentModel() {
        return this.currentModel;
    }
}

document.addEventListener('DOMContentLoaded', () => {
    window.llmManager = new LLMManager();
});