/**
 * LLM Manager - Manage AI model selection
 */

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
            background: #10b981;
            color: white;
            padding: 8px 16px;
            border-radius: 10px;
            z-index: 10001;
            font-size: 13px;
            animation: slideUp 0.3s ease, fadeOut 0.3s ease 1.7s forwards;
        `;
        document.body.appendChild(toast);
        setTimeout(() => toast.remove(), 2000);
    }

    getCurrentModel() {
        return this.currentModel;
    }
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        window.llmManager = new LLMManager();
    });
} else {
    window.llmManager = new LLMManager();
}