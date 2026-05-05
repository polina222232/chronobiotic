/**
 * Language Switcher - 9 languages support with instant UI update
 */

const translations = {
    en: {
        new_chat: 'New chat',
        settings: 'Settings',
        ready: 'Ready',
        placeholder: 'Ask me about chronobiotics...',
        recent: 'Recent chats',
        search: 'Search conversations...',
        agent_type: 'Agent Type',
        citation_style: 'Citation Style',
        streaming: 'Streaming responses',
        show_citations: 'Show citations',
        voice: 'Voice',
        voice_language: 'Language',
        voice_gender: 'Voice Gender',
        auto_play: 'Auto-play responses',
        thinking: 'AI is thinking...',
        no_chats: 'No chats yet',
        clear: 'Clear history',
        upload: 'Upload file',
        voice_input: 'Voice input',
        sound_effects: 'Sound effects',
        auto_scroll: 'Auto-scroll'
    },
    ru: {
        new_chat: 'Новый чат',
        settings: 'Настройки',
        ready: 'Готов',
        placeholder: 'Спросите о хронобиотиках...',
        recent: 'Недавние чаты',
        search: 'Поиск диалогов...',
        agent_type: 'Тип агента',
        citation_style: 'Стиль цитирования',
        streaming: 'Потоковые ответы',
        show_citations: 'Показывать цитаты',
        voice: 'Голос',
        voice_language: 'Язык',
        voice_gender: 'Пол голоса',
        auto_play: 'Автовоспроизведение',
        thinking: 'ИИ думает...',
        no_chats: 'Нет чатов',
        clear: 'Очистить историю',
        upload: 'Загрузить файл',
        voice_input: 'Голосовой ввод',
        sound_effects: 'Звуковые эффекты',
        auto_scroll: 'Автопрокрутка'
    },
    es: {
        new_chat: 'Nuevo chat',
        settings: 'Ajustes',
        ready: 'Listo',
        placeholder: 'Pregunta sobre cronobióticos...',
        recent: 'Charlas recientes',
        search: 'Buscar conversaciones...',
        agent_type: 'Tipo de agente',
        citation_style: 'Estilo de cita',
        streaming: 'Respuestas en streaming',
        show_citations: 'Mostrar citas',
        voice: 'Voz',
        voice_language: 'Idioma',
        voice_gender: 'Género de voz',
        auto_play: 'Reproducción automática',
        thinking: 'IA pensando...',
        no_chats: 'Sin chats',
        clear: 'Borrar historial',
        upload: 'Subir archivo',
        voice_input: 'Entrada de voz',
        sound_effects: 'Efectos de sonido',
        auto_scroll: 'Auto-desplazamiento'
    },
    fr: {
        new_chat: 'Nouveau chat',
        settings: 'Paramètres',
        ready: 'Prêt',
        placeholder: 'Demandez-moi sur les chronobiotiques...',
        recent: 'Chats récents',
        search: 'Rechercher...',
        agent_type: "Type d'agent",
        citation_style: 'Style de citation',
        streaming: 'Réponses en streaming',
        show_citations: 'Afficher les citations',
        voice: 'Voix',
        voice_language: 'Langue',
        voice_gender: 'Genre vocal',
        auto_play: 'Lecture automatique',
        thinking: 'IA réfléchit...',
        no_chats: 'Aucun chat',
        clear: 'Effacer l\'historique',
        upload: 'Télécharger',
        voice_input: 'Commande vocale',
        sound_effects: 'Effets sonores',
        auto_scroll: 'Défilement auto'
    },
    zh: {
        new_chat: '新对话',
        settings: '设置',
        ready: '就绪',
        placeholder: '询问关于时间生物学药物...',
        recent: '最近对话',
        search: '搜索对话...',
        agent_type: '代理类型',
        citation_style: '引用风格',
        streaming: '流式响应',
        show_citations: '显示引用',
        voice: '语音',
        voice_language: '语言',
        voice_gender: '语音性别',
        auto_play: '自动播放',
        thinking: 'AI思考中...',
        no_chats: '暂无对话',
        clear: '清除历史',
        upload: '上传文件',
        voice_input: '语音输入',
        sound_effects: '音效',
        auto_scroll: '自动滚动'
    },
    it: {
        new_chat: 'Nuovo chat',
        settings: 'Impostazioni',
        ready: 'Pronto',
        placeholder: 'Chiedimi sui cronobiotici...',
        recent: 'Chat recenti',
        search: 'Cerca conversazioni...',
        agent_type: 'Tipo di agente',
        citation_style: 'Stile citazione',
        streaming: 'Risposte in streaming',
        show_citations: 'Mostra citazioni',
        voice: 'Voce',
        voice_language: 'Lingua',
        voice_gender: 'Genere vocale',
        auto_play: 'Riproduzione automatica',
        thinking: 'IA pensando...',
        no_chats: 'Nessuna chat',
        clear: 'Cancella cronologia',
        upload: 'Carica file',
        voice_input: 'Input vocale',
        sound_effects: 'Effetti sonori',
        auto_scroll: 'Scorrimento auto'
    },
    ko: {
        new_chat: '새 채팅',
        settings: '설정',
        ready: '준비됨',
        placeholder: '크로노바이오틱스에 대해 물어보세요...',
        recent: '최근 채팅',
        search: '대화 검색...',
        agent_type: '에이전트 유형',
        citation_style: '인용 스타일',
        streaming: '스트리밍 응답',
        show_citations: '인용 표시',
        voice: '음성',
        voice_language: '언어',
        voice_gender: '음성 성별',
        auto_play: '자동 재생',
        thinking: 'AI 생각 중...',
        no_chats: '채팅 없음',
        clear: '기록 지우기',
        upload: '파일 업로드',
        voice_input: '음성 입력',
        sound_effects: '사운드 효과',
        auto_scroll: '자동 스크롤'
    },
    de: {
        new_chat: 'Neuer Chat',
        settings: 'Einstellungen',
        ready: 'Bereit',
        placeholder: 'Fragen Sie mich zu Chronobiotika...',
        recent: 'Letzte Chats',
        search: 'Unterhaltungen suchen...',
        agent_type: 'Agententyp',
        citation_style: 'Zitierstil',
        streaming: 'Streaming-Antworten',
        show_citations: 'Zitate anzeigen',
        voice: 'Stimme',
        voice_language: 'Sprache',
        voice_gender: 'Stimmgeschlecht',
        auto_play: 'Automatische Wiedergabe',
        thinking: 'KI denkt...',
        no_chats: 'Keine Chats',
        clear: 'Verlauf löschen',
        upload: 'Datei hochladen',
        voice_input: 'Spracheingabe',
        sound_effects: 'Soundeffekte',
        auto_scroll: 'Auto-Scroll'
    },
    hi: {
        new_chat: 'नई चैट',
        settings: 'सेटिंग्स',
        ready: 'तैयार',
        placeholder: 'मुझसे क्रोनोबायोटिक्स के बारे में पूछें...',
        recent: 'हाल की चैट',
        search: 'बातचीत खोजें...',
        agent_type: 'एजेंट प्रकार',
        citation_style: 'उद्धरण शैली',
        streaming: 'स्ट्रीमिंग प्रतिक्रियाएं',
        show_citations: 'उद्धरण दिखाएं',
        voice: 'आवाज',
        voice_language: 'भाषा',
        voice_gender: 'आवाज लिंग',
        auto_play: 'स्वत: चलाएं',
        thinking: 'एआई सोच रहा है...',
        no_chats: 'कोई चैट नहीं',
        clear: 'इतिहास साफ़ करें',
        upload: 'फ़ाइल अपलोड करें',
        voice_input: 'आवाज इनपुट',
        sound_effects: 'ध्वनि प्रभाव',
        auto_scroll: 'स्वत: स्क्रॉल'
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
        this.updateLanguageButton();
    }

    setupDropdown() {
        const langBtn = document.getElementById('langBtn');
        const langDropdown = document.getElementById('langDropdown');

        if (!langBtn || !langDropdown) {
            console.warn('Language button or dropdown not found');
            return;
        }

        // Remove existing listeners to avoid duplicates
        const newLangBtn = langBtn.cloneNode(true);
        langBtn.parentNode.replaceChild(newLangBtn, langBtn);

        newLangBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            if (langDropdown.style.display === 'none' || langDropdown.style.display === '') {
                langDropdown.style.display = 'block';
            } else {
                langDropdown.style.display = 'none';
            }
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

    setLanguage(lang) {
        if (!translations[lang]) return;

        this.currentLang = lang;
        localStorage.setItem('language', lang);
        const t = translations[lang];

        // Update elements with data-i18n attribute
        document.querySelectorAll('[data-i18n]').forEach(el => {
            const key = el.dataset.i18n;
            if (t[key]) el.textContent = t[key];
        });

        // Update elements with data-i18n-placeholder
        document.querySelectorAll('[data-i18n-placeholder]').forEach(el => {
            const key = el.dataset.i18nPlaceholder;
            if (t[key]) el.placeholder = t[key];
        });

        // Update specific UI elements
        const newChatBtn = document.getElementById('newChatBtn');
        if (newChatBtn) {
            const svg = newChatBtn.querySelector('svg');
            if (svg) {
                newChatBtn.innerHTML = svg.outerHTML + ` ${t.new_chat}`;
            } else {
                newChatBtn.innerHTML = `➕ ${t.new_chat}`;
            }
        }

        const settingsToggle = document.getElementById('settingsToggleBtn');
        if (settingsToggle) {
            const svg = settingsToggle.querySelector('svg');
            if (svg) {
                settingsToggle.innerHTML = svg.outerHTML + ` ${t.settings}`;
            } else {
                settingsToggle.innerHTML = `⚙️ ${t.settings}`;
            }
        }

        const voiceSettingsToggle = document.getElementById('voiceSettingsToggle');
        if (voiceSettingsToggle) {
            const svg = voiceSettingsToggle.querySelector('svg');
            if (svg) {
                voiceSettingsToggle.innerHTML = svg.outerHTML + ` ${t.voice}`;
            } else {
                voiceSettingsToggle.innerHTML = `🎤 ${t.voice}`;
            }
        }

        const statusText = document.getElementById('statusText');
        if (statusText) statusText.textContent = t.ready;

        const messageInput = document.getElementById('messageInput');
        if (messageInput) messageInput.placeholder = t.placeholder;

        const historyHeader = document.querySelector('.history-header span');
        if (historyHeader) historyHeader.textContent = t.recent;

        const historySearch = document.getElementById('historySearch');
        if (historySearch) historySearch.placeholder = t.search;

        const thinkingText = document.querySelector('.typing-text');
        if (thinkingText) thinkingText.textContent = t.thinking;

        // Update empty history message
        const emptyHistory = document.querySelector('.history-item.empty');
        if (emptyHistory) emptyHistory.textContent = t.no_chats;

        this.updateLanguageButton();

        // Dispatch event for other components
        document.dispatchEvent(new CustomEvent('languageChanged', { detail: { lang: lang } }));

        console.log(`Language changed to: ${lang}`);
    }

    updateLanguageButton() {
        const langBtn = document.getElementById('langBtn');
        if (langBtn) {
            const flags = {
                en: '🇺🇸', ru: '🇷🇺', es: '🇪🇸', fr: '🇫🇷',
                zh: '🇨🇳', it: '🇮🇹', ko: '🇰🇷', de: '🇩🇪', hi: '🇮🇳'
            };
            langBtn.textContent = flags[this.currentLang] || '🌐';
        }
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