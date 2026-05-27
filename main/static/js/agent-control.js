/**
 * Agent Control - Settings and preferences management
 */

(function() {
    document.addEventListener('DOMContentLoaded', function() {
        console.log('AgentControl initializing...');

        // ========== SETTINGS PANEL ==========
        const settingsToggle = document.getElementById('settingsToggleBtn');
        const settingsPanel = document.getElementById('settingsPanel');

        if (settingsToggle && settingsPanel) {
            settingsToggle.addEventListener('click', function(e) {
                e.preventDefault();
                e.stopPropagation();
                if (settingsPanel.style.display === 'none' || settingsPanel.style.display === '') {
                    settingsPanel.style.display = 'block';
                    const voicePanel = document.getElementById('voicePanel');
                    if (voicePanel) voicePanel.style.display = 'none';
                } else {
                    settingsPanel.style.display = 'none';
                }
            });
        }

        // ========== VOICE SETTINGS PANEL ==========
        const voiceSettingsToggle = document.getElementById('voiceSettingsToggle');
        const voicePanel = document.getElementById('voicePanel');

        if (voiceSettingsToggle && voicePanel) {
            voiceSettingsToggle.addEventListener('click', function(e) {
                e.preventDefault();
                e.stopPropagation();
                if (voicePanel.style.display === 'none' || voicePanel.style.display === '') {
                    voicePanel.style.display = 'block';
                    if (settingsPanel) settingsPanel.style.display = 'none';
                } else {
                    voicePanel.style.display = 'none';
                }
            });
        }

        // Close panels when clicking outside
        document.addEventListener('click', function(e) {
            if (settingsPanel && settingsPanel.style.display === 'block') {
                if (!settingsToggle.contains(e.target) && !settingsPanel.contains(e.target)) {
                    settingsPanel.style.display = 'none';
                }
            }
            if (voicePanel && voicePanel.style.display === 'block') {
                if (!voiceSettingsToggle.contains(e.target) && !voicePanel.contains(e.target)) {
                    voicePanel.style.display = 'none';
                }
            }
        });

        // ========== AGENT TYPE BUTTONS ==========
        const agentOptions = document.querySelectorAll('.agent-option');
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

                const agentNames = { assistant: 'Assistant', analyst: 'Analyst', researcher: 'Researcher' };
                const statusText = document.getElementById('statusText');
                if (statusText) statusText.textContent = agentNames[agent] || 'Ready';

                showToast(`Switched to ${this.textContent.trim()} mode`);
            });
        });

        // ========== CITATION STYLE BUTTONS ==========
        const citationOptions = document.querySelectorAll('.citation-option');
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

                if (window.citationManager) {
                    window.citationManager.setStyle(style);
                }
                showToast(`Citation style: ${this.textContent.trim()}`);
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
            });
        }

        // ========== CHECKBOXES ==========
        const streamToggle = document.getElementById('streamToggle');
        if (streamToggle) {
            streamToggle.checked = localStorage.getItem('streamResponse') !== 'false';
            streamToggle.addEventListener('change', function() {
                localStorage.setItem('streamResponse', this.checked);
                showToast(`Streaming: ${this.checked ? 'ON' : 'OFF'}`);
            });
        }

        const citationsToggle = document.getElementById('citationsToggle');
        if (citationsToggle) {
            citationsToggle.checked = localStorage.getItem('showCitations') !== 'false';
            citationsToggle.addEventListener('change', function() {
                localStorage.setItem('showCitations', this.checked);
                showToast(`Citations: ${this.checked ? 'ON' : 'OFF'}`);
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
                showToast(`Sound effects: ${this.checked ? 'ON' : 'OFF'}`);
            });
        }

        const autoScrollToggle = document.getElementById('autoScrollToggle');
        if (autoScrollToggle) {
            autoScrollToggle.checked = localStorage.getItem('autoScroll') !== 'false';
            autoScrollToggle.addEventListener('change', function() {
                localStorage.setItem('autoScroll', this.checked);
                showToast(`Auto-scroll: ${this.checked ? 'ON' : 'OFF'}`);
            });
        }

        // ========== VOICE SETTINGS ==========
        const voiceLangSelect = document.getElementById('voiceLangSelect');
        if (voiceLangSelect) {
            const savedLang = localStorage.getItem('voiceLanguage') || 'en-US';
            voiceLangSelect.value = savedLang;
            voiceLangSelect.addEventListener('change', function() {
                localStorage.setItem('voiceLanguage', this.value);
                showToast(`Voice language: ${this.options[this.selectedIndex].text}`);
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
            });
        }

        const autoPlayVoice = document.getElementById('autoPlayVoiceCheck');
        if (autoPlayVoice) {
            autoPlayVoice.checked = localStorage.getItem('autoPlayVoice') !== 'false';
            autoPlayVoice.addEventListener('change', function() {
                localStorage.setItem('autoPlayVoice', this.checked);
                showToast(`Auto-play voice: ${this.checked ? 'ON' : 'OFF'}`);
            });
        }

        // ========== GENDER BUTTONS ==========
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
                showToast(`Voice: ${this.textContent.trim()}`);
            });
        });

        // ========== TEST VOICE BUTTON ==========
        const testVoiceBtn = document.getElementById('testVoiceBtn');
        if (testVoiceBtn) {
            testVoiceBtn.addEventListener('click', function() {
                const lang = voiceLangSelect?.value || 'en-US';
                const rate = parseFloat(voiceSpeedSlider?.value || '1');
                const pitch = parseFloat(voicePitchSlider?.value || '1');

                const testText = lang.startsWith('ru') ? 'Привет! Это тест голосовой системы.' :
                                lang.startsWith('es') ? '¡Hola! Esta es una prueba del sistema de voz.' :
                                lang.startsWith('fr') ? 'Bonjour ! Ceci est un test du système vocal.' :
                                lang.startsWith('zh') ? '你好！这是语音系统的测试。' :
                                'Hello! This is a test of the voice system.';

                const utterance = new SpeechSynthesisUtterance(testText);
                utterance.lang = lang;
                utterance.rate = rate;
                utterance.pitch = pitch;

                window.speechSynthesis.cancel();
                window.speechSynthesis.speak(utterance);
                showToast('🔊 Testing voice...');
            });
        }

        function showToast(message) {
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
                font-weight: 500;
                animation: slideUp 0.3s ease, fadeOut 0.3s ease 1.7s forwards;
            `;
            document.body.appendChild(toast);
            setTimeout(() => toast.remove(), 2000);
        }

        console.log('AgentControl initialized');
    });
})();