/**
 * Agent Control - Manages agent types and behavior
 */

class AgentControl {
    constructor() {
        this.agentType = localStorage.getItem('agentType') || 'assistant';
        this.init();
    }

    init() {
        this.setupAgentType();
        this.updateStatus();
    }

    setupAgentType() {
        const select = document.getElementById('agentTypeSelect');
        if (select) {
            select.value = this.agentType;
            select.addEventListener('change', (e) => {
                this.setAgentType(e.target.value);
            });
        }
    }

    setAgentType(type) {
        this.agentType = type;
        localStorage.setItem('agentType', type);
        this.updateStatus();

        const prompts = {
            assistant: "You are a helpful assistant for the Chronobiotics Database. Provide clear, accurate information about chronobiotics, their mechanisms, targets, and effects.",
            analyst: "You are a data analyst specializing in chronobiotics. Provide structured insights, compare data, identify patterns, and present findings in a clear, organized manner.",
            researcher: "You are a scientific researcher in chronobiology. Provide detailed explanations, cite relevant research, discuss molecular mechanisms, and suggest research directions."
        };

        console.log(`Agent switched to ${type} mode`);

        if (window.agentChat) {
            window.agentChat.showToast(`Switched to ${type.charAt(0).toUpperCase() + type.slice(1)} mode`);
        }
    }

    updateStatus() {
        const statusText = document.getElementById('statusText');
        const lang = localStorage.getItem('language') || 'en';

        const names = {
            assistant: { en: 'Assistant', ru: 'Ассистент', es: 'Asistente', fr: 'Assistant', zh: '助手', it: 'Assistente', ko: '어시스턴트', de: 'Assistent', hi: 'सहायक' },
            analyst: { en: 'Analyst', ru: 'Аналитик', es: 'Analista', fr: 'Analyste', zh: '分析师', it: 'Analista', ko: '분석가', de: 'Analytiker', hi: 'विश्लेषक' },
            researcher: { en: 'Researcher', ru: 'Исследователь', es: 'Investigador', fr: 'Chercheur', zh: '研究员', it: 'Ricercatore', ko: '연구원', de: 'Forscher', hi: 'शोधकर्ता' }
        };

        if (statusText) {
            statusText.textContent = names[this.agentType]?.[lang] || names[this.agentType]?.en || 'Ready';
        }
    }

    getAgentType() {
        return this.agentType;
    }
}

const agentControl = new AgentControl();
window.agentControl = agentControl;