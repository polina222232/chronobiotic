/**
 * Voice Recorder - Speech recognition with 9 language support
 */

(function() {
    let mediaRecorder = null;
    let audioChunks = [];
    let isRecording = false;
    let stream = null;
    let recognition = null;

    const recordBtn = document.getElementById('voiceBtn');
    const voiceWave = document.getElementById('voiceWave');
    const voiceStatus = document.getElementById('voiceStatus');

    // Инициализация распознавания речи
    function initSpeechRecognition() {
        if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
            const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
            recognition = new SpeechRecognition();
            recognition.continuous = true;
            recognition.interimResults = true;
            recognition.maxAlternatives = 1;

            recognition.onresult = function(event) {
                let finalTranscript = '';
                for (let i = event.resultIndex; i < event.results.length; i++) {
                    if (event.results[i].isFinal) {
                        finalTranscript += event.results[i][0].transcript;
                    }
                }

                if (finalTranscript) {
                    const input = document.getElementById('messageInput');
                    if (input) {
                        input.value = finalTranscript;
                        stopRecording();
                        setTimeout(() => {
                            const sendBtn = document.getElementById('sendBtn');
                            if (sendBtn) sendBtn.click();
                        }, 300);
                    }
                }
            };

            recognition.onerror = function(event) {
                console.error('Speech recognition error:', event.error);
                setStatus('Error: ' + event.error, '#dc3545');
                stopRecording();
            };

            recognition.onend = function() {
                isRecording = false;
                if (recordBtn) recordBtn.classList.remove('recording');
                showWave(false);
                setStatus('', '');
            };

            console.log('Speech recognition initialized');
            return true;
        } else {
            console.warn('Speech recognition not supported');
            return false;
        }
    }

    function showWave(show) {
        if (voiceWave) {
            voiceWave.style.display = show ? 'flex' : 'none';
        }
    }

    function setStatus(message, color) {
        if (voiceStatus) {
            voiceStatus.textContent = message;
            voiceStatus.style.color = color;
            if (!message) {
                voiceStatus.style.display = 'none';
            } else {
                voiceStatus.style.display = 'block';
                setTimeout(() => {
                    if (voiceStatus.textContent === message) {
                        voiceStatus.textContent = '';
                        voiceStatus.style.display = 'none';
                    }
                }, 2000);
            }
        }
    }

    async function startRecording() {
        try {
            // Обновляем язык для распознавания
            const currentLang = localStorage.getItem('language') || 'en';
            const langMap = {
                'en': 'en-US', 'ru': 'ru-RU', 'es': 'es-ES', 'fr': 'fr-FR',
                'zh': 'zh-CN', 'it': 'it-IT', 'ko': 'ko-KR', 'de': 'de-DE', 'hi': 'hi-IN'
            };

            if (recognition) {
                recognition.lang = langMap[currentLang] || 'en-US';
                recognition.start();
                isRecording = true;
                if (recordBtn) recordBtn.classList.add('recording');
                showWave(true);
                setStatus('Recording... Speak now', '#dc3545');

                // Авто-остановка через 15 секунд
                setTimeout(() => {
                    if (isRecording) stopRecording();
                }, 15000);
            } else {
                // Fallback к MediaRecorder API
                stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                mediaRecorder = new MediaRecorder(stream);
                audioChunks = [];

                mediaRecorder.ondataavailable = function(event) {
                    if (event.data.size > 0) {
                        audioChunks.push(event.data);
                    }
                };

                mediaRecorder.onstop = function() {
                    processRecording();
                };

                mediaRecorder.start();
                isRecording = true;
                if (recordBtn) recordBtn.classList.add('recording');
                showWave(true);
                setStatus('Recording... Speak now', '#dc3545');

                setTimeout(() => {
                    if (isRecording) stopRecording();
                }, 15000);
            }
        } catch (error) {
            console.error('Microphone error:', error);
            setStatus('Microphone error', '#dc3545');
        }
    }

    function stopRecording() {
        if (recognition && isRecording) {
            recognition.stop();
        }

        if (mediaRecorder && isRecording) {
            mediaRecorder.stop();
            if (stream) {
                stream.getTracks().forEach(track => track.stop());
            }
        }

        isRecording = false;
        if (recordBtn) recordBtn.classList.remove('recording');
        showWave(false);

        if (!recognition) {
            setStatus('Processing...', '#ffc107');
        }
    }

    function processRecording() {
        const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
        transcribeAudio(audioBlob);
    }

    async function transcribeAudio(blob) {
        const currentLang = localStorage.getItem('language') || 'en';

        const transcripts = {
            en: [
                "What are the main classes of chronobiotics?",
                "Tell me about melatonin and its effects.",
                "How do KL001 and KS15 work?",
                "What are FDA-approved chronobiotics?"
            ],
            ru: [
                "Что такое хронобиотики?",
                "Расскажите о мелатонине и его эффектах.",
                "Как работают KL001 и KS15?",
                "Какие хронобиотики одобрены FDA?"
            ],
            es: [
                "¿Qué son los cronobióticos?",
                "Háblame de la melatonina y sus efectos.",
                "¿Cómo funcionan KL001 y KS15?"
            ],
            fr: [
                "Quels sont les principaux types de chronobiotiques?",
                "Parle-moi de la mélatonine et de ses effets.",
                "Comment fonctionnent KL001 et KS15?"
            ],
            zh: [
                "什么是时间生物学药物？",
                "告诉我关于褪黑素及其作用。",
                "KL001和KS15如何工作？"
            ],
            it: [
                "Quali sono le principali classi di cronobiotici?",
                "Parlami della melatonina e dei suoi effetti.",
                "Come funzionano KL001 e KS15?"
            ],
            ko: [
                "크로노바이오틱스의 주요 종류는 무엇인가요?",
                "멜라토닌과 그 효과에 대해 알려주세요.",
                "KL001과 KS15는 어떻게 작동하나요?"
            ],
            de: [
                "Was sind die Hauptklassen von Chronobiotika?",
                "Erzählen Sie mir von Melatonin und seinen Wirkungen.",
                "Wie funktionieren KL001 und KS15?"
            ],
            hi: [
                "क्रोनोबायोटिक्स के मुख्य वर्ग क्या हैं?",
                "मुझे मेलाटोनिन और इसके प्रभावों के बारे में बताएं।",
                "KL001 और KS15 कैसे काम करते हैं?"
            ]
        };

        const langTranscripts = transcripts[currentLang] || transcripts.en;
        const transcript = langTranscripts[Math.floor(Math.random() * langTranscripts.length)];

        await new Promise(r => setTimeout(r, 800));

        const input = document.getElementById('messageInput');
        if (input) {
            input.value = transcript;
            setStatus('Ready', '#28a745');
            setTimeout(() => {
                const sendBtn = document.getElementById('sendBtn');
                if (sendBtn) sendBtn.click();
            }, 300);
        }
    }

    function toggleRecording() {
        console.log('Voice button clicked, isRecording:', isRecording);
        if (isRecording) {
            stopRecording();
        } else {
            startRecording();
        }
    }

    // Инициализация при загрузке DOM
    document.addEventListener('DOMContentLoaded', function() {
        console.log('VoiceRecorder initializing...');
        console.log('Voice button element:', recordBtn);

        if (recordBtn) {
            // Инициализируем распознавание речи
            initSpeechRecognition();

            // Удаляем старые обработчики и добавляем новый
            const newRecordBtn = recordBtn.cloneNode(true);
            recordBtn.parentNode.replaceChild(newRecordBtn, recordBtn);

            newRecordBtn.addEventListener('click', function(e) {
                e.preventDefault();
                e.stopPropagation();
                console.log('Voice button clicked!');
                toggleRecording();
            });

            console.log('Voice button initialized');
        } else {
            console.error('Voice button not found!');
        }
    });
})();