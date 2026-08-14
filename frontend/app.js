/* ═══════════════════════════════════════════════════════════════
   VoiceRAG — Frontend Application Logic
   ═══════════════════════════════════════════════════════════════ */

(function () {
    'use strict';

    // ── API Configuration ─────────────────────────────────────
    const API_BASE = window.location.origin;

    // ── DOM Elements ──────────────────────────────────────────
    const micBtn = document.getElementById('micBtn');
    const micStatus = document.getElementById('micStatus');
    const textInput = document.getElementById('textInput');
    const sendBtn = document.getElementById('sendBtn');
    const languageSelect = document.getElementById('language');
    const resultsArea = document.getElementById('resultsArea');
    const transcriptCard = document.getElementById('transcriptCard');
    const transcriptText = document.getElementById('transcriptText');
    const transcriptLang = document.getElementById('transcriptLang');
    const loadingCard = document.getElementById('loadingCard');
    const answerCard = document.getElementById('answerCard');
    const answerText = document.getElementById('answerText');
    const groundedBadge = document.getElementById('groundedBadge');
    const guardrailCard = document.getElementById('guardrailCard');
    const guardrailList = document.getElementById('guardrailList');
    const sourcesCard = document.getElementById('sourcesCard');
    const sourcesList = document.getElementById('sourcesList');
    const latencyCard = document.getElementById('latencyCard');
    const latencyBars = document.getElementById('latencyBars');

    // ── State ─────────────────────────────────────────────────
    let isRecording = false;
    let mediaRecorder = null;
    let audioChunks = [];

    // ── Language Display Names ────────────────────────────────
    const LANG_NAMES = {
        'en': 'English',
        'hi': 'हिन्दी',
        'ta': 'தமிழ்',
        'bn': 'বাংলা',
        'te': 'తెలుగు',
        'mr': 'मराठी',
        'unknown': 'Unknown',
    };

    // ── Microphone Recording ──────────────────────────────────

    async function startRecording() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    sampleRate: 16000,
                    channelCount: 1,
                    echoCancellation: true,
                    noiseSuppression: true,
                }
            });

            audioChunks = [];
            mediaRecorder = new MediaRecorder(stream, {
                mimeType: MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
                    ? 'audio/webm;codecs=opus'
                    : 'audio/webm'
            });

            mediaRecorder.ondataavailable = (e) => {
                if (e.data.size > 0) audioChunks.push(e.data);
            };

            mediaRecorder.onstop = async () => {
                stream.getTracks().forEach(t => t.stop());
                const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });

                // Convert to WAV for Sarvam API compatibility
                const wavBlob = await convertToWav(audioBlob);
                await submitVoiceQuery(wavBlob);
            };

            mediaRecorder.start(100); // collect in 100ms chunks
            isRecording = true;
            micBtn.classList.add('recording');
            micStatus.textContent = 'Recording... Click to stop';

        } catch (err) {
            console.error('Microphone error:', err);
            micStatus.textContent = 'Microphone access denied';
            setTimeout(() => { micStatus.textContent = 'Click to speak'; }, 3000);
        }
    }

    function stopRecording() {
        if (mediaRecorder && isRecording) {
            mediaRecorder.stop();
            isRecording = false;
            micBtn.classList.remove('recording');
            micStatus.textContent = 'Processing...';
        }
    }

    // ── Audio Conversion (WebM → WAV) ─────────────────────────

    async function convertToWav(webmBlob) {
        const audioContext = new (window.AudioContext || window.webkitAudioContext)({
            sampleRate: 16000
        });

        const arrayBuffer = await webmBlob.arrayBuffer();

        try {
            const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
            const wavData = audioBufferToWav(audioBuffer);
            audioContext.close();
            return new Blob([wavData], { type: 'audio/wav' });
        } catch (e) {
            console.warn('WAV conversion failed, sending WebM:', e);
            audioContext.close();
            return webmBlob;
        }
    }

    function audioBufferToWav(buffer) {
        const numChannels = 1;
        const sampleRate = buffer.sampleRate;
        const format = 1; // PCM
        const bitDepth = 16;

        const channelData = buffer.getChannelData(0);
        const samples = new Int16Array(channelData.length);

        for (let i = 0; i < channelData.length; i++) {
            const s = Math.max(-1, Math.min(1, channelData[i]));
            samples[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
        }

        const dataLength = samples.length * 2;
        const headerLength = 44;
        const wavBuffer = new ArrayBuffer(headerLength + dataLength);
        const view = new DataView(wavBuffer);

        // WAV header
        writeString(view, 0, 'RIFF');
        view.setUint32(4, 36 + dataLength, true);
        writeString(view, 8, 'WAVE');
        writeString(view, 12, 'fmt ');
        view.setUint32(16, 16, true);
        view.setUint16(20, format, true);
        view.setUint16(22, numChannels, true);
        view.setUint32(24, sampleRate, true);
        view.setUint32(28, sampleRate * numChannels * bitDepth / 8, true);
        view.setUint16(32, numChannels * bitDepth / 8, true);
        view.setUint16(34, bitDepth, true);
        writeString(view, 36, 'data');
        view.setUint32(40, dataLength, true);

        // Write samples
        const offset = 44;
        for (let i = 0; i < samples.length; i++) {
            view.setInt16(offset + i * 2, samples[i], true);
        }

        return wavBuffer;
    }

    function writeString(view, offset, str) {
        for (let i = 0; i < str.length; i++) {
            view.setUint8(offset + i, str.charCodeAt(i));
        }
    }

    // ── API Calls ─────────────────────────────────────────────

    async function submitVoiceQuery(audioBlob) {
        showLoading();

        const formData = new FormData();
        formData.append('audio', audioBlob, 'recording.wav');

        const lang = languageSelect.value;
        if (lang) formData.append('language', lang);

        try {
            const response = await fetch(`${API_BASE}/query`, {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) throw new Error(`HTTP ${response.status}`);

            const data = await response.json();
            displayResults(data);
        } catch (err) {
            console.error('Query error:', err);
            showError('Failed to process query. Please try again.');
        }
    }

    async function submitTextQuery(query) {
        if (!query.trim()) return;

        showLoading();
        textInput.disabled = true;
        sendBtn.disabled = true;

        const lang = languageSelect.value || null;

        try {
            const response = await fetch(`${API_BASE}/query/text`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query: query.trim(), language: lang }),
            });

            if (!response.ok) throw new Error(`HTTP ${response.status}`);

            const data = await response.json();
            displayResults(data);
        } catch (err) {
            console.error('Query error:', err);
            showError('Failed to process query. Please try again.');
        } finally {
            textInput.disabled = false;
            sendBtn.disabled = false;
        }
    }

    // ── Display Results ───────────────────────────────────────

    function showLoading() {
        resultsArea.classList.remove('hidden');
        hideAllCards();
        loadingCard.classList.remove('hidden');
    }

    function hideAllCards() {
        [transcriptCard, loadingCard, answerCard, guardrailCard, sourcesCard, latencyCard]
            .forEach(c => c.classList.add('hidden'));
    }

    function showError(message) {
        hideAllCards();
        answerCard.classList.remove('hidden');
        answerText.textContent = message;
        groundedBadge.classList.add('hidden');
        micStatus.textContent = 'Click to speak';
    }

    function displayResults(data) {
        hideAllCards();
        micStatus.textContent = 'Click to speak';

        // Transcript
        if (data.transcript) {
            transcriptCard.classList.remove('hidden');
            transcriptText.textContent = data.transcript;
            transcriptLang.textContent = LANG_NAMES[data.detected_language] || data.detected_language;
        }

        // Answer
        if (data.answer) {
            answerCard.classList.remove('hidden');
            answerText.textContent = data.answer;

            // Grounding badge
            groundedBadge.classList.remove('hidden');
            if (data.grounded) {
                groundedBadge.textContent = '✓ Grounded';
                groundedBadge.classList.remove('ungrounded');
            } else {
                groundedBadge.textContent = '⚠ Ungrounded';
                groundedBadge.classList.add('ungrounded');
            }
        }

        // Guardrail flags
        if (data.guardrail_flags && data.guardrail_flags.length > 0) {
            guardrailCard.classList.remove('hidden');
            guardrailList.innerHTML = '';
            data.guardrail_flags.forEach(flag => {
                const li = document.createElement('li');
                li.textContent = `⚠ ${flag}`;
                guardrailList.appendChild(li);
            });
        }

        // Sources
        if (data.cited_chunks && data.cited_chunks.length > 0) {
            sourcesCard.classList.remove('hidden');
            sourcesList.innerHTML = '';
            data.cited_chunks.forEach((chunk, i) => {
                const item = document.createElement('div');
                item.className = 'source-item';
                item.innerHTML = `
                    <div class="source-header">
                        <span class="source-id">Passage ${i + 1} [${chunk.language}]</span>
                        <span class="source-score">Score: ${chunk.score.toFixed(4)}</span>
                    </div>
                    <p class="source-text">${escapeHtml(chunk.text)}</p>
                `;
                sourcesList.appendChild(item);
            });
        }

        // Latency
        if (data.latency) {
            latencyCard.classList.remove('hidden');
            renderLatencyBars(data.latency);
        }
    }

    function renderLatencyBars(latency) {
        const stages = [
            { key: 'stt_ms', label: 'STT', cls: 'stt' },
            { key: 'retrieval_ms', label: 'Retrieval', cls: 'retrieval' },
            { key: 'generation_ms', label: 'LLM Gen', cls: 'generation' },
            { key: 'guardrails_ms', label: 'Guards', cls: 'guardrails' },
            { key: 'total_ms', label: 'Total', cls: 'total' },
        ];

        const maxMs = Math.max(latency.total_ms || 1, 1);
        latencyBars.innerHTML = '';

        stages.forEach(stage => {
            const ms = latency[stage.key] || 0;
            if (ms === 0 && stage.key !== 'total_ms') return;

            const pct = Math.min((ms / maxMs) * 100, 100);

            const row = document.createElement('div');
            row.className = 'latency-row';
            row.innerHTML = `
                <span class="latency-label">${stage.label}</span>
                <div class="latency-bar-container">
                    <div class="latency-bar ${stage.cls}" style="width: ${pct}%"></div>
                </div>
                <span class="latency-value">${ms.toFixed(0)} ms</span>
            `;
            latencyBars.appendChild(row);
        });
    }

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    // ── Event Listeners ───────────────────────────────────────

    micBtn.addEventListener('click', () => {
        if (isRecording) {
            stopRecording();
        } else {
            startRecording();
        }
    });

    sendBtn.addEventListener('click', () => {
        submitTextQuery(textInput.value);
    });

    textInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            submitTextQuery(textInput.value);
        }
    });

    // ── Health Check on Load ──────────────────────────────────

    async function checkHealth() {
        try {
            const response = await fetch(`${API_BASE}/health`);
            const data = await response.json();
            if (data.status === 'ok') {
                console.log('API healthy:', data);
            }
        } catch (e) {
            console.warn('API health check failed:', e);
        }
    }

    checkHealth();

})();
