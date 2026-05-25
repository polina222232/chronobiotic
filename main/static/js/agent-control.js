

(function() {
    // Ждем полной загрузки DOM
    document.addEventListener('DOMContentLoaded', function() {
        console.log('AgentControl initializing...');

        // ========== SETTINGS PANEL ==========
        const settingsBtn = document.getElementById('settingsBtn');
        const settingsPanel = document.getElementById('settingsPanel');

        if (settingsBtn && settingsPanel) {
            console.log('Settings button found');
            settingsBtn.addEventListener('click', function(e) {
                e.preventDefault();
                e.stopPropagation();
                console.log('Settings button clicked');
                if (settingsPanel.style.display === 'none' || settingsPanel.style.display === '') {
                    settingsPanel.style.display = 'block';
                    // Закрываем voice панель если открыта
                    const voicePanel = document.getElementById('voicePanel');
                    if (voicePanel) voicePanel.style.display = 'none';
                } else {
                    settingsPanel.style.display = 'none';
                }
            });
        } else {
            console.error('Settings button or panel not found!');
        }

        // ========== VOICE SETTINGS PANEL ==========
        const voiceSettingsBtn = document.getElementById('voiceSettingsBtn');
        const voicePanel = document.getElementById('voicePanel');

        if (voiceSettingsBtn && voicePanel) {
            console.log('Voice settings button found');
            voiceSettingsBtn.addEventListener('click', function(e) {
                e.preventDefault();
                e.stopPropagation();
                console.log('Voice settings button clicked');
                if (voicePanel.style.display === 'none' || voicePanel.style.display === '') {
                    voicePanel.style.display = 'block';
                    // Закрываем settings панель если открыта
                    if (settingsPanel) settingsPanel.style.display = 'none';
                } else {
                    voicePanel.style.display = 'none';
                }
            });
        } else {
            console.error('Voice settings button or panel not found!');
        }

        // Закрытие панелей при клике вне их
        document.addEventListener('click', function(e) {
            if (settingsPanel && settingsPanel.style.display === 'block') {
                if (!settingsBtn.contains(e.target) && !settingsPanel.contains(e.target)) {
                    settingsPanel.style.display = 'none';
                }
            }
            if (voicePanel && voicePanel.style.display === 'block') {
                if (!voiceSettingsBtn.contains(e.target) && !voicePanel.contains(e.target)) {
                    voicePanel.style.display = 'none';
                }
            }
        });

        // ========== AGENT TYPE BUTTONS ==========
        const agentOptions = document.querySelectorAll('.agent-option');
        console.log('Agent options found:', agentOptions.length);

        // Загрузка сохраненного типа агента
        const savedAgent = localStorage.getItem('agentType') || 'assistant';

        agentOptions.forEach(btn => {
            if (btn.dataset.agent === savedAgent) {
                btn.classList.add('active');
            }
            btn.addEventListener('click', function() {
                agentOptions.forEach(b => b.classList.remove('active'));
                this.classList.add('active');
                const agent = this.dataset.agent;
                localStorage.setItem('agentType', agent);
                console.log('Agent changed to:', agent);

                // Обновляем статус
                const statusText = document.getElementById('statusText');
                if (statusText) {
                    const names = { assistant: 'Assistant', analyst: 'Analyst', researcher: 'Researcher' };
                    statusText.textContent = names[agent] || 'Ready';
                }

                // Показываем уведомление
                showToast(`Switched to ${this.textContent.trim()} mode`);
            });
        });

        // ========== CITATION STYLE BUTTONS ==========
        const citationOptions = document.querySelectorAll('.citation-option');
        console.log('Citation options found:', citationOptions.length);

        // Загрузка сохраненного стиля цитирования
        const savedStyle = localStorage.getItem('citationStyle') || 'gost-r';

        citationOptions.forEach(btn => {
            if (btn.dataset.style === savedStyle) {
                btn.classList.add('active');
            }
            btn.addEventListener('click', function() {
                citationOptions.forEach(b => b.classList.remove('active'));
                this.classList.add('active');
                const style = this.dataset.style;
                localStorage.setItem('citationStyle', style);
                console.log('Citation style changed to:', style);

                if (window.citationManager) {
                    window.citationManager.setStyle(style);
                }

                showToast(`Citation style changed to ${this.textContent.trim()}`);
            });
        });

        // ========== TEMPERATURE SLIDER ==========
        const tempSlider = document.getElementById('tempSlider');
        const tempVal = document.getElementById('tempVal');

        if (tempSlider && tempVal) {
            const savedTemp = localStorage.getItem('temperature');
            if (savedTemp) {
                tempSlider.value = savedTemp;
                tempVal.textContent = savedTemp;
            }
            tempSlider.addEventListener('input', function() {
                tempVal.textContent = this.value;
                localStorage.setItem('temperature', this.value);
                console.log('Temperature changed to:', this.value);
            });
        }

        // ========== CHECKBOXES ==========
        const streamToggle = document.getElementById('streamToggle');
        if (streamToggle) {
            streamToggle.checked = localStorage.getItem('streamResponse') !== 'false';
            streamToggle.addEventListener('change', function() {
                localStorage.setItem('streamResponse', this.checked);
                console.log('Stream response:', this.checked);
            });
        }

        const citationsToggle = document.getElementById('citationsToggle');
        if (citationsToggle) {
            citationsToggle.checked = localStorage.getItem('showCitations') !== 'false';
            citationsToggle.addEventListener('change', function() {
                localStorage.setItem('showCitations', this.checked);
                console.log('Show citations:', this.checked);
            });
        }

        const soundToggle = document.getElementById('soundToggle');
        if (soundToggle) {
            soundToggle.checked = localStorage.getItem('soundEffects') !== 'false';
            soundToggle.addEventListener('change', function() {
                localStorage.setItem('soundEffects', this.checked);
                if (window.audioEffects) {
                    window.audioEffects.toggle(this.checked);
                }
                console.log('Sound effects:', this.checked);
            });
        }

        const autoScrollToggle = document.getElementById('autoScrollToggle');
        if (autoScrollToggle) {
            autoScrollToggle.checked = localStorage.getItem('autoScroll') !== 'false';
            autoScrollToggle.addEventListener('change', function() {
                localStorage.setItem('autoScroll', this.checked);
                console.log('Auto-scroll:', this.checked);
            });
        }

        // ========== VOICE SETTINGS ==========
        const voiceLangSelect = document.getElementById('voiceLangSelect');
        if (voiceLangSelect) {
            const savedLang = localStorage.getItem('voiceLanguage') || 'en-US';
            voiceLangSelect.value = savedLang;
            voiceLangSelect.addEventListener('change', function() {
                localStorage.setItem('voiceLanguage', this.value);
                console.log('Voice language:', this.value);
            });
        }

        const voiceSpeedSlider = document.getElementById('voiceSpeedSlider');
        const voiceSpeedVal = document.getElementById('voiceSpeedVal');
        if (voiceSpeedSlider && voiceSpeedVal) {
            const savedSpeed = localStorage.getItem('voiceSpeed') || '1';
            voiceSpeedSlider.value = savedSpeed;
            voiceSpeedVal.textContent = savedSpeed;
            voiceSpeedSlider.addEventListener('input', function() {
                voiceSpeedVal.textContent = this.value;
                localStorage.setItem('voiceSpeed', this.value);
                console.log('Voice speed:', this.value);
            });
        }

        const voicePitchSlider = document.getElementById('voicePitchSlider');
        const voicePitchVal = document.getElementById('voicePitchVal');
        if (voicePitchSlider && voicePitchVal) {
            const savedPitch = localStorage.getItem('voicePitch') || '1';
            voicePitchSlider.value = savedPitch;
            voicePitchVal.textContent = savedPitch;
            voicePitchSlider.addEventListener('input', function() {
                voicePitchVal.textContent = this.value;
                localStorage.setItem('voicePitch', this.value);
                console.log('Voice pitch:', this.value);
            });
        }

        const autoPlayVoice = document.getElementById('autoPlayVoiceCheck');
        if (autoPlayVoice) {
            autoPlayVoice.checked = localStorage.getItem('autoPlayVoice') !== 'false';
            autoPlayVoice.addEventListener('change', function() {
                localStorage.setItem('autoPlayVoice', this.checked);
                console.log('Auto-play voice:', this.checked);
            });
        }

        // GENDER BUTTONS
        const genderOptions = document.querySelectorAll('.gender-option');
        const savedGender = localStorage.getItem('voiceGender') || 'female';

        genderOptions.forEach(btn => {
            if (btn.dataset.gender === savedGender) {
                btn.classList.add('active');
            }
            btn.addEventListener('click', function() {
                genderOptions.forEach(b => b.classList.remove('active'));
                this.classList.add('active');
                const gender = this.dataset.gender;
                localStorage.setItem('voiceGender', gender);
                console.log('Voice gender:', gender);
                showToast(`Voice set to ${this.textContent.trim()}`);
            });
        });

        // TEST VOICE BUTTON
        const testVoiceBtn = document.getElementById('testVoiceBtn');
        if (testVoiceBtn) {
            testVoiceBtn.addEventListener('click', function() {
                const lang = voiceLangSelect?.value || 'en-US';
                const rate = parseFloat(voiceSpeedSlider?.value || '1');
                const pitch = parseFloat(voicePitchSlider?.value || '1');

                const utterance = new SpeechSynthesisUtterance('Hello! This is a test of the voice system.');
                utterance.lang = lang;
                utterance.rate = rate;
                utterance.pitch = pitch;

                window.speechSynthesis.cancel();
                window.speechSynthesis.speak(utterance);

                showToast('Testing voice...');
            });
        }

        // Helper function
        function showToast(message) {
            const toast = document.createElement('div');
            toast.textContent = message;
            toast.style.cssText = `
                position: fixed;
                bottom: 20px;
                left: 50%;
                transform: translateX(-50%);
                background: #333;
                color: white;
                padding: 8px 16px;
                border-radius: 8px;
                z-index: 10000;
                font-size: 13px;
            `;
            document.body.appendChild(toast);
            setTimeout(() => toast.remove(), 2000);
        }

        console.log('AgentControl initialized successfully');
    });
})();