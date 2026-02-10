/**
 * Language Switcher - 9 languages support
 */

const translations = {
    en: {
        'new_chat': 'New chat',
        'settings': 'Settings',
        'ready': 'Ready',
        'placeholder': 'Ask me about chronobiotics...',
        'recent': 'Recent chats',
        'no_chats': 'No chats yet',
        'clear': 'Clear history',
        'upload': 'Upload file',
        'voice': 'Voice input'
    },
    ru: {
        'new_chat': 'Новый чат',
        'settings': 'Настройки',
        'ready': 'Готов',
        'placeholder': 'Спросите о хронобиотиках...',
        'recent': 'Недавние чаты',
        'no_chats': 'Нет чатов',
        'clear': 'Очистить историю',
        'upload': 'Загрузить файл',
        'voice': 'Голосовой ввод'
    },
    es: {
        'new_chat': 'Nuevo chat',
        'settings': 'Ajustes',
        'ready': 'Listo',
        'placeholder': 'Pregunta sobre cronobióticos...',
        'recent': 'Charlas recientes',
        'no_chats': 'Sin chats',
        'clear': 'Borrar historial',
        'upload': 'Subir archivo',
        'voice': 'Entrada de voz'
    },
    fr: {
        'new_chat': 'Nouveau chat',
        'settings': 'Paramètres',
        'ready': 'Prêt',
        'placeholder': 'Demandez-moi sur les chronobiotiques...',
        'recent': 'Chats récents',
        'no_chats': 'Aucun chat',
        'clear': 'Effacer l\'historique',
        'upload': 'Télécharger',
        'voice': 'Commande vocale'
    },
    zh: {
        'new_chat': '新对话',
        'settings': '设置',
        'ready': '就绪',
        'placeholder': '询问关于时间生物学药物...',
        'recent': '最近对话',
        'no_chats': '暂无对话',
        'clear': '清除历史',
        'upload': '上传文件',
        'voice': '语音输入'
    },
    it: {
        'new_chat': 'Nuovo chat',
        'settings': 'Impostazioni',
        'ready': 'Pronto',
        'placeholder': 'Chiedimi sui cronobiotici...',
        'recent': 'Chat recenti',
        'no_chats': 'Nessuna chat',
        'clear': 'Cancella cronologia',
        'upload': 'Carica file',
        'voice': 'Input vocale'
    },
    ko: {
        'new_chat': '새 채팅',
        'settings': '설정',
        'ready': '준비됨',
        'placeholder': '크로노바이오틱스에 대해 물어보세요...',
        'recent': '최근 채팅',
        'no_chats': '채팅 없음',
        'clear': '기록 지우기',
        'upload': '파일 업로드',
        'voice': '음성 입력'
    },
    de: {
        'new_chat': 'Neuer Chat',
        'settings': 'Einstellungen',
        'ready': 'Bereit',
        'placeholder': 'Fragen Sie mich zu Chronobiotika...',
        'recent': 'Letzte Chats',
        'no_chats': 'Keine Chats',
        'clear': 'Verlauf löschen',
        'upload': 'Datei hochladen',
        'voice': 'Spracheingabe'
    },
    hi: {
        'new_chat': 'नई चैट',
        'settings': 'सेटिंग्स',
        'ready': 'तैयार',
        'placeholder': 'मुझसे क्रोनोबायोटिक्स के बारे में पूछें...',
        'recent': 'हाल की चैट',
        'no_chats': 'कोई चैट नहीं',
        'clear': 'इतिहास साफ़ करें',
        'upload': 'फ़ाइल अपलोड करें',
        'voice': 'आवाज इनपुट'
    }
};

class LanguageSwitcher {
    constructor() {
        this.currentLang = localStorage.getItem('language') || 'en';
        this.init();
    }

    init() {
        this.setLanguage(this.currentLang);
        this.setupDropdown();
    }

    setupDropdown() {
        const langBtn = document.getElementById('langBtn');
        const langDropdown = document.getElementById('langDropdown');

        if (langBtn && langDropdown) {
            langBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                langDropdown.style.display = langDropdown.style.display === 'none' ? 'block' : 'none';
            });

            const options = langDropdown.querySelectorAll('div');
            options.forEach(opt => {
                opt.addEventListener('click', () => {
                    const lang = opt.dataset.lang;
                    this.setLanguage(lang);
                    langDropdown.style.display = 'none';
                });
            });

            document.addEventListener('click', () => {
                langDropdown.style.display = 'none';
            });
        }
    }

    setLanguage(lang) {
        if (!translations[lang]) return;

        this.currentLang = lang;
        localStorage.setItem('language', lang);

        const t = translations[lang];

        // Update new chat button
        const newChatBtn = document.getElementById('newChatBtn');
        if (newChatBtn) {
            newChatBtn.innerHTML = `➕ ${t.new_chat}`;
        }

        // Update settings button
        const settingsBtn = document.getElementById('settingsToggleBtn');
        if (settingsBtn) {
            settingsBtn.innerHTML = `⚙️ ${t.settings}`;
        }

        // Update status
        const statusText = document.getElementById('statusText');
        if (statusText) statusText.textContent = t.ready;

        // Update placeholder
        const input = document.getElementById('messageInput');
        if (input) input.placeholder = t.placeholder;

        // Update history header
        const historyHeader = document.querySelector('.history-header');
        if (historyHeader) {
            historyHeader.innerHTML = `📋 ${t.recent}`;
        }

        // Update clear button title
        const clearBtn = document.getElementById('clearHistoryBtn');
        if (clearBtn) clearBtn.title = t.clear;

        // Update file button title
        const fileBtn = document.getElementById('fileBtn');
        if (fileBtn) fileBtn.title = t.upload;

        // Update voice button title
        const voiceBtn = document.getElementById('voiceBtn');
        if (voiceBtn) voiceBtn.title = t.voice;

        // Update language button
        const langBtn = document.getElementById('langBtn');
        if (langBtn) {
            const flags = { en: '🇺🇸', ru: '🇷🇺', es: '🇪🇸', fr: '🇫🇷', zh: '🇨🇳', it: '🇮🇹', ko: '🇰🇷', de: '🇩🇪', hi: '🇮🇳' };
            langBtn.textContent = flags[lang];
        }

        // Update empty history message
        const emptyHistory = document.querySelector('.empty-history');
        if (emptyHistory) {
            emptyHistory.textContent = t.no_chats;
        }

        console.log(`Language changed to: ${lang}`);
    }

    getCurrentLanguage() {
        return this.currentLang;
    }

    translate(key) {
        return translations[this.currentLang]?.[key] || translations.en[key] || key;
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.langSwitcher = new LanguageSwitcher();
});