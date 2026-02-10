/**
 * Language Switcher Module
 * Handles multi-language support
 */

class LanguageSwitcher {
    constructor() {
        this.currentLang = localStorage.getItem('language') || 'en';
        this.translations = this.getTranslations();
        this.init();
    }

    init() {
        this.applyLanguage(this.currentLang);
    }

    getTranslations() {
        return {
            en: {
                'new_chat': 'New chat',
                'settings': 'Settings',
                'voice_settings': 'Voice settings',
                'search_placeholder': 'Search conversations...',
                'message_placeholder': 'Ask me anything about chronobiotics...',
                'send': 'Send',
                'thinking': 'AI is thinking...',
                'copy': 'Copy',
                'listen': 'Listen',
                'regenerate': 'Regenerate',
                'clear_history': 'Clear history',
                'export_chat': 'Export chat',
                'online': 'Online',
                'ready': 'Ready',
                'processing': 'Processing...'
            },
            ru: {
                'new_chat': 'Новый чат',
                'settings': 'Настройки',
                'voice_settings': 'Настройки голоса',
                'search_placeholder': 'Поиск диалогов...',
                'message_placeholder': 'Спросите меня о хронобиотиках...',
                'send': 'Отправить',
                'thinking': 'ИИ думает...',
                'copy': 'Копировать',
                'listen': 'Слушать',
                'regenerate': 'Сгенерировать снова',
                'clear_history': 'Очистить историю',
                'export_chat': 'Экспорт чата',
                'online': 'В сети',
                'ready': 'Готов',
                'processing': 'Обработка...'
            },
            es: {
                'new_chat': 'Nuevo chat',
                'settings': 'Ajustes',
                'voice_settings': 'Ajustes de voz',
                'search_placeholder': 'Buscar conversaciones...',
                'message_placeholder': 'Pregúntame sobre cronobióticos...',
                'send': 'Enviar',
                'thinking': 'IA pensando...',
                'copy': 'Copiar',
                'listen': 'Escuchar',
                'regenerate': 'Regenerar',
                'clear_history': 'Borrar historial',
                'export_chat': 'Exportar chat',
                'online': 'En línea',
                'ready': 'Listo',
                'processing': 'Procesando...'
            }
        };
    }

    applyLanguage(lang) {
        const translation = this.translations[lang] || this.translations.en;

        // Update text content for elements with data-i18n attribute
        document.querySelectorAll('[data-i18n]').forEach(element => {
            const key = element.dataset.i18n;
            if (translation[key]) {
                if (element.tagName === 'INPUT' || element.tagName === 'TEXTAREA') {
                    element.placeholder = translation[key];
                } else {
                    element.textContent = translation[key];
                }
            }
        });

        // Update message input placeholder
        const messageInput = document.getElementById('messageInput');
        if (messageInput) {
            messageInput.placeholder = translation['message_placeholder'];
        }

        // Update history search placeholder
        const historySearch = document.getElementById('historySearch');
        if (historySearch) {
            historySearch.placeholder = translation['search_placeholder'];
        }

        // Update new chat button
        const newChatBtn = document.getElementById('newChatBtn');
        if (newChatBtn) {
            const icon = newChatBtn.querySelector('i');
            if (icon) {
                newChatBtn.innerHTML = '';
                newChatBtn.appendChild(icon);
                newChatBtn.appendChild(document.createTextNode(' ' + translation['new_chat']));
            }
        }

        localStorage.setItem('language', lang);
    }

    switchTo(lang) {
        if (this.translations[lang]) {
            this.currentLang = lang;
            this.applyLanguage(lang);
            return true;
        }
        return false;
    }

    getCurrentLanguage() {
        return this.currentLang;
    }
}

const languageSwitcher = new LanguageSwitcher();