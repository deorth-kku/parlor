function $$(id) {
  return document.getElementById(id);
}
let video, inputSourceSelect, inputSourceSelectMobile, languageSelect, languageSelectMobile, voiceSelect, voiceSelectMobile, messagesDiv, statusEl;
let viewportWrap, viewportGlow, waveformCanvas, waveformCtx, mirrorToggleBtn;
let mirrorToggleBoundEl = null;
function refreshElements() {
  video = $$('video');
  inputSourceSelect = $$('inputSourceSelect');
  inputSourceSelectMobile = $$('inputSourceSelectMobile');
  languageSelect = $$('languageSelect');
  languageSelectMobile = $$('languageSelectMobile');
  voiceSelect = $$('voiceSelect');
  voiceSelectMobile = $$('voiceSelectMobile');
  messagesDiv = $$('messages');
  statusEl = $$('status');
  viewportWrap = $$('viewportWrap');
  viewportGlow = viewportWrap ? viewportWrap.querySelector('.viewport-glow') : null;
  waveformCanvas = $$('waveform');
  waveformCtx = waveformCanvas.getContext('2d');
  mirrorToggleBtn = $$('mirrorToggle');
  syncMirrorToggleBinding();
  if (video) {
    video.srcObject = mediaStream || null;
    video.classList.toggle('mirror', currentMirrorState);
  }
  applyVideoLayout();
  updateMirrorToggleUI();
  updateStateUI();
}

let ws, mediaStream, myvad, audioStream;
let cameraEnabled = true;
let audioCtx, currentSource;
let state = 'loading';
let ignoreIncomingAudio = false;
let activeAssistantMsg = null;
let ttsOptions = null;
let selectedLanguage = 'ja';
let selectedVoice = 'jf_alpha';
let activeTurnId = null;
let interruptedTurnId = null;
let listeningEnabled = true;
let vadRunning = false;
let wakeLockSentinel = null;
let wakeLockRetryTimer = 0;

// Streaming audio playback state
let streamSampleRate = 24000;
let streamNextTime = 0;
let streamSources = [];
let streamTtsTime = null;
let streamPlaybackStarted = false;
let scrollRaf = 0;

// ── Waveform Visualizer ──
let analyser;
let micSource = null;

function getMicSource() {
  if (!micSource && audioStream && audioStream.getAudioTracks().length) {
    micSource = audioCtx.createMediaStreamSource(audioStream);
  }
  return micSource;
}

function scrollMessagesToBottom(immediate = false) {
  if (!messagesDiv) return;
  if (immediate) {
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
    return;
  }
  if (scrollRaf) return;
  scrollRaf = requestAnimationFrame(() => {
    scrollRaf = 0;
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
  });
}
const BAR_COUNT = 40;
const BAR_GAP = 3;
const CAPTURE_MAX_WIDTH = 1280;
let waveformRAF;
let waveformResizeRAF = 0;
let ambientPhase = 0;

function initWaveformCanvas() {
  if (!waveformCanvas || !waveformCtx) return;
  const dpr = window.devicePixelRatio || 1;
  const rect = waveformCanvas.getBoundingClientRect();
  if (!rect.width || !rect.height) return;
  waveformCtx.setTransform(1, 0, 0, 1, 0, 0);
  waveformCanvas.width = Math.round(rect.width * dpr);
  waveformCanvas.height = Math.round(rect.height * dpr);
  waveformCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

function scheduleWaveformCanvasResize() {
  if (waveformResizeRAF) cancelAnimationFrame(waveformResizeRAF);
  waveformResizeRAF = requestAnimationFrame(() => {
    waveformResizeRAF = 0;
    initWaveformCanvas();
  });
}

function getStateColor() {
  const colors = { listening: '#4ade80', processing: '#f59e0b', speaking: '#818cf8', loading: '#3a3d46' };
  return colors[state] || colors.loading;
}

function saveTtsSelection(languageCode, voice) {
  try {
    localStorage.setItem('tts_language', languageCode);
    localStorage.setItem('tts_voice', voice);
  } catch {}
}

function renderLanguageOptions() {
  for (const select of [languageSelect, languageSelectMobile]) {
    if (!select) continue;
    select.innerHTML = '';
    for (const language of ttsOptions.languages) {
      const option = document.createElement('option');
      option.value = language.code;
      option.textContent = `${language.label} (${language.code})`;
      select.appendChild(option);
    }
  }
}

function renderVoiceOptions(languageCode, preferredVoice = null) {
  const language = ttsOptions.languages.find(item => item.code === languageCode) || ttsOptions.languages[0];
  const chosen = language.voices.includes(preferredVoice) ? preferredVoice : language.default_voice;
  for (const select of [voiceSelect, voiceSelectMobile]) {
    if (!select) continue;
    select.innerHTML = '';
    for (const voice of language.voices) {
      const option = document.createElement('option');
      option.value = voice;
      option.textContent = voice;
      select.appendChild(option);
    }
    select.disabled = false;
    select.value = chosen;
  }
  selectedVoice = chosen;
}

function loadTtsSelection() {
  try {
    const lang = localStorage.getItem('tts_language');
    const voice = localStorage.getItem('tts_voice');
    if (lang && voice) return { languageCode: lang, voice };
  } catch {}
  return null;
}

function setTtsSelection(languageCode, voice) {
  selectedLanguage = languageCode;
  languageSelect.value = languageCode;
  languageSelectMobile.value = languageCode;
  renderVoiceOptions(languageCode, voice);
  saveTtsSelection(languageCode, voice);
}

function drawWaveform() {
  const w = waveformCanvas.getBoundingClientRect().width;
  const h = waveformCanvas.getBoundingClientRect().height;
  waveformCtx.clearRect(0, 0, w, h);

  const barWidth = (w - (BAR_COUNT - 1) * BAR_GAP) / BAR_COUNT;
  const color = getStateColor();
  waveformCtx.fillStyle = color;

  let dataArray = null;
  if (analyser) {
    dataArray = new Uint8Array(analyser.frequencyBinCount);
    analyser.getByteFrequencyData(dataArray);
  }

  for (let i = 0; i < BAR_COUNT; i++) {
    let amplitude;
    if (dataArray) {
      const binIndex = Math.floor((i / BAR_COUNT) * dataArray.length * 0.6);
      amplitude = dataArray[binIndex] / 255;
    }

    if (!dataArray || amplitude < 0.02) {
      ambientPhase += 0.0001;
      const drift = Math.sin(ambientPhase * 3 + i * 0.4) * 0.5 + 0.5;
      amplitude = 0.03 + drift * 0.04;
    }

    const barH = Math.max(2, amplitude * (h - 4));
    const x = i * (barWidth + BAR_GAP);
    const y = (h - barH) / 2;
    const r = Math.max(0, Math.min(barWidth / 2, barH / 2, 3));

    waveformCtx.globalAlpha = 0.3 + amplitude * 0.7;
    waveformCtx.beginPath();
    waveformCtx.roundRect(x, y, Math.max(0, barWidth), Math.max(0, barH), r);
    waveformCtx.fill();
  }

  waveformCtx.globalAlpha = 1;
  waveformRAF = requestAnimationFrame(drawWaveform);
}

// ── Dynamic glow intensity for speaking state ──
function updateSpeakingGlow() {
  if (!viewportWrap || !viewportGlow) return;
  if (state !== 'speaking' || !analyser) return;
  const data = new Uint8Array(analyser.frequencyBinCount);
  analyser.getByteFrequencyData(data);
  let sum = 0;
  for (let i = 0; i < data.length; i++) sum += data[i];
  const avg = sum / data.length / 255;
  const intensity = 0.3 + avg * 0.7;
  viewportWrap.style.setProperty('--speak-intensity', intensity);
  const spread = 20 + avg * 60;
  const inner = 15 + avg * 25;
  viewportGlow.style.boxShadow =
    `0 0 ${spread}px ${spread * 0.4}px rgba(129,140,248,${intensity * 0.25})`;
  viewportWrap.style.boxShadow =
    `inset 0 0 ${inner}px rgba(129,140,248,${intensity * 0.15}), 0 0 ${inner}px rgba(129,140,248,${intensity * 0.2})`;
  requestAnimationFrame(updateSpeakingGlow);
}

// ── State Machine ──
function setState(newState) {
  state = newState;
  updateStateUI();
  syncWakeLock();
}

function getDisplayedState() {
  if (!listeningEnabled && state === 'listening') return 'muted';
  return state;
}

function updateStateUI() {
  if (!viewportWrap) return;
  const displayedState = getDisplayedState();

  viewportWrap.className = `viewport-wrap ${state}${displayedState === 'muted' ? ' paused' : ''}`;

  if (statusEl) {
    const statusLabels = {
      loading: 'Loading...',
      listening: listeningEnabled ? 'Listening' : 'Listening Off',
      muted: 'Listening Off',
      processing: 'Processing',
      speaking: 'Speaking',
      disconnected: 'Disconnected'
    };
    // Action word shown only on hover (stop an in-flight turn).
    const statusActions = {
      processing: 'Stop',
    };
    const statusClasses = {
      loading: 'loading',
      listening: listeningEnabled ? 'listening' : 'muted',
      muted: 'muted',
      processing: 'processing',
      speaking: 'speaking',
      disconnected: 'disconnected'
    };
    const label = statusLabels[displayedState] || displayedState;
    const action = statusActions[displayedState];
    const cls = statusClasses[displayedState] || '';
    statusEl.className = `status-pill ${cls}`;
    if (displayedState === 'processing') statusEl.classList.add('stoppable');
    statusEl.innerHTML =
      `<span class="label-base">${label}</span>` +
      (action ? `<span class="label-action">${action}</span>` : '');
  }

  if (state !== 'speaking') {
    viewportWrap.style.boxShadow = '';
    if (viewportGlow) viewportGlow.style.boxShadow = '';
  }

  const stateVars = {
    listening: ['#4ade80', 'rgba(74,222,128,0.12)'],
    muted: ['#f87171', 'rgba(248,113,113,0.12)'],
    processing: ['#f59e0b', 'rgba(245,158,11,0.12)'],
    speaking: ['#818cf8', 'rgba(129,140,248,0.12)'],
    loading: ['#3a3d46', 'rgba(58,61,70,0.12)'],
    disconnected: ['#f87171', 'rgba(248,113,113,0.12)'],
  };
  const [glow, glowDim] = stateVars[displayedState] || stateVars.loading;
  document.documentElement.style.setProperty('--glow', glow);
  document.documentElement.style.setProperty('--glow-dim', glowDim);

  if (state === 'speaking') requestAnimationFrame(updateSpeakingGlow);

  if (myvad) {
    myvad.setOptions({ positiveSpeechThreshold: state === 'speaking' ? 0.92 : 0.5 });
  }

  if (displayedState === 'listening' && analyser) {
    try { getMicSource().connect(analyser); } catch {}
  } else {
    try { getMicSource().disconnect(analyser); } catch {}
  }
}

// ── WebSocket ──
function connect() {
  ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`);
  ws.onopen = () => {
    if (state !== 'loading') setState('listening');
  };
  ws.onclose = () => {
    setState('disconnected');
    setTimeout(connect, 2000);
  };
  ws.onmessage = ({ data }) => {
    const msg = JSON.parse(data);
    if (msg.type === 'text') {
      if (msg.transcription) {
        updateLatestUserMessage(msg.transcription);
      }
      activeAssistantMsg = addMessage('assistant', msg.text, `LLM ${msg.llm_time}s`);
    } else if (msg.type === 'text_start') {
      activeTurnId = msg.turn_id || null;
      ignoreIncomingAudio = false;
      activeAssistantMsg = addMessage('assistant', '', '');
    } else if (msg.type === 'transcription_delta') {
      if (msg.turn_id && activeTurnId && msg.turn_id !== activeTurnId) return;
      updateLatestUserMessage(msg.text);
    } else if (msg.type === 'text_delta') {
      if (msg.turn_id && activeTurnId && msg.turn_id !== activeTurnId) return;
      if (activeAssistantMsg) {
        const meta = activeAssistantMsg.querySelector('.meta');
        const current = activeAssistantMsg.childNodes[0];
        if (current && current.nodeType === Node.TEXT_NODE) {
          current.textContent += msg.delta;
        } else {
          activeAssistantMsg.insertBefore(document.createTextNode(msg.delta), meta || null);
        }
        scrollMessagesToBottom();
      } else {
        activeAssistantMsg = addMessage('assistant', msg.delta, '');
      }
    } else if (msg.type === 'text_replace') {
      if (msg.turn_id && activeTurnId && msg.turn_id !== activeTurnId) return;
      if (activeAssistantMsg) {
        const meta = activeAssistantMsg.querySelector('.meta');
        const textNode = activeAssistantMsg.childNodes[0];
        if (textNode && textNode.nodeType === Node.TEXT_NODE) {
          textNode.textContent = msg.text;
        } else {
          activeAssistantMsg.insertBefore(document.createTextNode(msg.text), meta || null);
        }
        scrollMessagesToBottom();
      } else {
        activeAssistantMsg = addMessage('assistant', msg.text, '');
      }
    } else if (msg.type === 'text_end') {
      if (msg.turn_id && activeTurnId && msg.turn_id !== activeTurnId) return;
      if (msg.transcription) {
        updateLatestUserMessage(msg.transcription);
      }
      if (activeAssistantMsg) {
        const meta = activeAssistantMsg.querySelector('.meta');
        const textNode = activeAssistantMsg.childNodes[0];
        if (textNode && textNode.nodeType === Node.TEXT_NODE) {
          textNode.textContent = msg.text;
        } else {
          activeAssistantMsg.insertBefore(document.createTextNode(msg.text), meta || null);
        }
        if (meta) meta.textContent = `LLM ${msg.llm_time}s`;
      } else {
        activeAssistantMsg = addMessage('assistant', msg.text, `LLM ${msg.llm_time}s`);
      }
      activeAssistantMsg = null;
      scrollMessagesToBottom(true);
      if (!streamPlaybackStarted && state === 'processing') {
        setState('listening');
        activeTurnId = null;
      }
    } else if (msg.type === 'turn_interrupted') {
      if (msg.turn_id && activeTurnId && msg.turn_id !== activeTurnId) return;
      // Finalize any half-rendered assistant bubble so it doesn't stay stuck
      // on the ellipsis placeholder.
      if (activeAssistantMsg) {
        const meta = activeAssistantMsg.querySelector('.meta');
        const textNode = activeAssistantMsg.childNodes[0];
        const hasText = textNode && textNode.nodeType === Node.TEXT_NODE && textNode.textContent.trim();
        if (!hasText) {
          const fresh = document.createTextNode('—');
          if (textNode) {
            activeAssistantMsg.replaceChild(fresh, textNode);
          } else {
            activeAssistantMsg.insertBefore(fresh, meta || null);
          }
        }
        if (meta) meta.textContent = 'Interrupted';
      }
      activeAssistantMsg = null;
      interruptedTurnId = null;
      ignoreIncomingAudio = false;
      stopPlayback();
      setState('listening');
      activeTurnId = null;
    } else if (msg.type === 'audio_start') {
      if (msg.turn_id && activeTurnId && msg.turn_id !== activeTurnId) return;
      if (interruptedTurnId && msg.turn_id === interruptedTurnId) return;
      if (ignoreIncomingAudio) return;
      streamSampleRate = msg.sample_rate || 24000;
      startStreamPlayback();
    } else if (msg.type === 'audio_chunk') {
      if (msg.turn_id && activeTurnId && msg.turn_id !== activeTurnId) return;
      if (interruptedTurnId && msg.turn_id === interruptedTurnId) return;
      if (ignoreIncomingAudio) return;
      queueAudioChunk(msg.audio);
    } else if (msg.type === 'audio_end') {
      if (msg.turn_id && activeTurnId && msg.turn_id !== activeTurnId) return;
      if (ignoreIncomingAudio) {
        ignoreIncomingAudio = false;
        stopPlayback();
        setState('listening');
        return;
      }
      streamTtsTime = msg.tts_time;
      syncStreamPlaybackState();
      const meta = messagesDiv.querySelector('.msg.assistant:last-child .meta');
      if (meta) meta.textContent += ` · TTS ${msg.tts_time}s`;
      if (!streamPlaybackStarted && state !== 'listening') {
        setState('listening');
      }
      interruptedTurnId = null;
      activeTurnId = null;
    }
  };
}

// ── Video Source Switching ──
const VIDEO_SOURCE_STORAGE_KEY = 'parlor-video-source';
let videoSource = 'off';
let preferredVideoSource = loadVideoSourceSelection();
let currentCameraId = null;
let screenShareTrack = null;
let suppressScreenShareEnded = false;
let videoDeviceLabels = new Map();
const MIRROR_OVERRIDE_STORAGE_KEY = 'parlor-mirror-overrides';
let mirrorOverrides = loadMirrorOverrides();
let currentMirrorState = false;

refreshElements();

function loadMirrorOverrides() {
  try {
    const raw = localStorage.getItem(MIRROR_OVERRIDE_STORAGE_KEY);
    if (!raw) return {};
    const parsed = JSON.parse(raw);
    return parsed && typeof parsed === 'object' ? parsed : {};
  } catch {
    return {};
  }
}

function loadVideoSourceSelection() {
  try {
    return localStorage.getItem(VIDEO_SOURCE_STORAGE_KEY) || 'off';
  } catch {
    return 'off';
  }
}

function saveVideoSourceSelection(source) {
  try {
    localStorage.setItem(VIDEO_SOURCE_STORAGE_KEY, source);
  } catch {}
}

function saveMirrorOverrides() {
  try {
    localStorage.setItem(MIRROR_OVERRIDE_STORAGE_KEY, JSON.stringify(mirrorOverrides));
  } catch {}
}

function getMirrorOverride(deviceId) {
  if (!deviceId) return null;
  return Object.prototype.hasOwnProperty.call(mirrorOverrides, deviceId) ? !!mirrorOverrides[deviceId] : null;
}

function isLikelyFrontCameraLabel(label) {
  const normalized = (label || '').trim().toLowerCase();
  if (!normalized) return false;
  if (/(back|rear|environment|world|后置|後置)/i.test(normalized)) return false;
  return /(front|user|facetime|true(depth)?|integrated|built-in|内置|內置|前置)/i.test(normalized);
}

function shouldMirrorCamera(stream, deviceId) {
  const track = stream?.getVideoTracks?.()[0];
  const facingMode = track?.getSettings?.().facingMode;
  if (facingMode === 'user') return true;
  if (facingMode === 'environment') return false;
  return isLikelyFrontCameraLabel(videoDeviceLabels.get(deviceId));
}

function updateMirrorToggleUI() {
  if (!mirrorToggleBtn) return;
  const isCameraSource = !!(videoSource && videoSource !== 'off' && videoSource !== 'screen');
  mirrorToggleBtn.hidden = !isCameraSource;
  if (!isCameraSource) return;
  mirrorToggleBtn.classList.toggle('off', !currentMirrorState);
  const stateEl = mirrorToggleBtn.querySelector('.state');
  if (stateEl) stateEl.textContent = currentMirrorState ? 'On' : 'Off';
  const override = getMirrorOverride(currentCameraId);
  mirrorToggleBtn.title = override === null
    ? 'Auto detected. Click to override mirror.'
    : 'Manual override enabled. Click to toggle mirror.';
}

function syncMirrorToggleBinding() {
  if (mirrorToggleBoundEl && mirrorToggleBoundEl !== mirrorToggleBtn) {
    mirrorToggleBoundEl.removeEventListener('click', toggleMirrorOverride);
  }
  if (mirrorToggleBtn && mirrorToggleBoundEl !== mirrorToggleBtn) {
    mirrorToggleBtn.addEventListener('click', toggleMirrorOverride);
  }
  mirrorToggleBoundEl = mirrorToggleBtn;
}

function setVideoMirror(shouldMirror) {
  currentMirrorState = !!shouldMirror;
  video.classList.toggle('mirror', currentMirrorState);
  updateMirrorToggleUI();
}

function resolveMirrorPreference(stream, deviceId) {
  const override = getMirrorOverride(deviceId);
  if (override !== null) return override;
  return shouldMirrorCamera(stream, deviceId);
}

function toggleMirrorOverride() {
  if (!currentCameraId) return;
  mirrorOverrides[currentCameraId] = !currentMirrorState;
  saveMirrorOverrides();
  setVideoMirror(mirrorOverrides[currentCameraId]);
}

function handleScreenShareEnded() {
  if (suppressScreenShareEnded) return;
  suppressScreenShareEnded = true;
  stopStream();
  videoSource = 'off';
  preferredVideoSource = 'off';
  saveVideoSourceSelection('off');
  inputSourceSelect.value = 'off';
  setVideoMirror(false);
  applyVideoLayout();
}

async function enumerateVideoDevices() {
  try {
    // Request camera permission first so labels are populated
    const probe = await navigator.mediaDevices.getUserMedia({ video: true });
    const devices = await navigator.mediaDevices.enumerateDevices();
    probe.getTracks().forEach(t => t.stop());
    const videoInputs = devices.filter(d => d.kind === 'videoinput');
    videoDeviceLabels = new Map(videoInputs.map(d => [d.deviceId, d.label || '']));
    populateVideoSelect(videoInputs);
  } catch (e) { console.warn('Enumerate devices failed:', e.message); }
}

function populateVideoSelect(videoInputs) {
  const selects = [inputSourceSelect, inputSourceSelectMobile].filter(Boolean);
  for (const select of selects) {
    select.innerHTML = '';
    const offOpt = document.createElement('option');
    offOpt.value = 'off';
    offOpt.textContent = 'Off';
    select.appendChild(offOpt);
    for (const device of videoInputs) {
      const opt = document.createElement('option');
      opt.value = device.deviceId;
      opt.textContent = device.label || 'Camera ' + (videoInputs.indexOf(device) + 1);
      select.appendChild(opt);
    }
    const screenOpt = document.createElement('option');
    screenOpt.value = 'screen';
    screenOpt.textContent = 'Screen Share';
    select.appendChild(screenOpt);
    if (select.querySelector('option[value="' + preferredVideoSource + '"]')) {
      select.value = preferredVideoSource;
    } else {
      preferredVideoSource = 'off';
      select.value = 'off';
      saveVideoSourceSelection('off');
    }
  }
}

async function stopStream() {
  suppressScreenShareEnded = true;
  if (mediaStream) {
    mediaStream.getTracks().forEach(t => t.stop());
    mediaStream = null;
  }
  screenShareTrack = null;
  video.srcObject = null;
  suppressScreenShareEnded = false;
  applyVideoLayout();
}

function applyVideoLayout() {
  const hasVideo = !!(videoSource && videoSource !== 'off' && mediaStream);
  document.body.classList.toggle('layout-no-video', !hasVideo);
  scheduleWaveformCanvasResize();
  syncWakeLock();
}

function shouldHoldWakeLock() {
  return document.visibilityState === 'visible' && videoSource === 'off';
}

function releaseWakeLock() {
  if (wakeLockRetryTimer) {
    clearTimeout(wakeLockRetryTimer);
    wakeLockRetryTimer = 0;
  }
  if (!wakeLockSentinel) return;
  const sentinel = wakeLockSentinel;
  wakeLockSentinel = null;
  try { sentinel.release(); } catch {}
}

async function requestWakeLock() {
  if (!shouldHoldWakeLock()) return false;
  if (!('wakeLock' in navigator) || wakeLockSentinel) return !!wakeLockSentinel;
  try {
    wakeLockSentinel = await navigator.wakeLock.request('screen');
    wakeLockSentinel.addEventListener('release', () => {
      wakeLockSentinel = null;
      if (shouldHoldWakeLock()) {
        scheduleWakeLockRetry();
      }
    });
    return true;
  } catch {
    wakeLockSentinel = null;
    return false;
  }
}

function scheduleWakeLockRetry() {
  if (wakeLockRetryTimer) return;
  wakeLockRetryTimer = window.setTimeout(() => {
    wakeLockRetryTimer = 0;
    requestWakeLock();
  }, 1000);
}

function syncWakeLock() {
  if (!shouldHoldWakeLock()) {
    releaseWakeLock();
    return;
  }
  requestWakeLock();
}

async function switchVideoSource(source, options = {}) {
  const { persist = true, force = false } = options;
  if (!force && source === videoSource && (source === 'off' || !!mediaStream)) return;
  if (persist) {
    preferredVideoSource = source;
    saveVideoSourceSelection(source);
  }
  await stopStream();
  videoSource = 'off';
  currentCameraId = null;
  setVideoMirror(false);
  if (source === 'off') {
    mediaStream = null;
  } else if (source === 'screen') {
    try {
      const stream = await navigator.mediaDevices.getDisplayMedia({ video: { cursor: 'always' } });
      mediaStream = stream; screenShareTrack = stream.getVideoTracks()[0] || null;
      if (screenShareTrack) screenShareTrack.onended = handleScreenShareEnded;
      video.srcObject = stream; videoSource = 'screen';
      currentCameraId = null;
      setVideoMirror(false);
    } catch (e) { console.warn('Screen share failed:', e.message); }
  } else {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { deviceId: { exact: source } } });
      mediaStream = stream; video.srcObject = stream; videoSource = source; currentCameraId = source;
      setVideoMirror(resolveMirrorPreference(stream, source));
    } catch (e) { console.warn('Camera start failed:', e.message); }
  }
  if (!persist) {
    preferredVideoSource = videoSource;
  }
  if (inputSourceSelect && inputSourceSelect.value !== videoSource) {
    inputSourceSelect.value = videoSource;
  }
  applyVideoLayout();
}


function captureFrame() {
  if (videoSource === 'off' || !video.videoWidth) return null;
  const canvas = document.createElement('canvas');
  const scale = video.videoWidth > CAPTURE_MAX_WIDTH ? CAPTURE_MAX_WIDTH / video.videoWidth : 1;
  canvas.width = Math.round(video.videoWidth * scale); canvas.height = Math.round(video.videoHeight * scale);
  canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
  return canvas.toDataURL('image/jpeg', 0.75).split(',')[1];
}

// ── VAD Handlers ──
let speakingStartedAt = 0;
const BARGE_IN_GRACE_MS = 800;

function handleSpeechStart() {
  if (!listeningEnabled) return;
  if (state === 'speaking') {
    if (Date.now() - speakingStartedAt < BARGE_IN_GRACE_MS) {
      console.log('Barge-in suppressed (echo grace period)');
      return;
    }
    stopPlayback();
    ignoreIncomingAudio = true;
    interruptedTurnId = activeTurnId;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'interrupt' }));
    }
    setState('listening');
    console.log('Barge-in: interrupted playback');
  }
}

function handleSpeechEnd(audio) {
  if (!listeningEnabled) return;
  if (state !== 'listening') return;
  if (!ws || ws.readyState !== WebSocket.OPEN) return;

  const wavBase64 = float32ToWavBase64(audio);
  const imageBase64 = captureFrame();

  setState('processing');
  addMessage('user', '<span class="loading-dots"><span></span><span></span><span></span></span>', imageBase64 ? 'with camera' : '', true);

  const payload = { audio: wavBase64, tts_language: selectedLanguage, tts_voice: selectedVoice };
  if (imageBase64) payload.image = imageBase64;
  ws.send(JSON.stringify(payload));
}

// ── Float32 @ 16kHz → WAV base64 ──
function float32ToWavBase64(samples) {
  const buf = new ArrayBuffer(44 + samples.length * 2);
  const v = new DataView(buf);
  const w = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
  w(0,'RIFF'); v.setUint32(4, 36 + samples.length * 2, true); w(8,'WAVE'); w(12,'fmt ');
  v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
  v.setUint32(24, 16000, true); v.setUint32(28, 32000, true); v.setUint16(32, 2, true);
  v.setUint16(34, 16, true); w(36,'data'); v.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    v.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
  }
  const bytes = new Uint8Array(buf);
  let bin = ''; for (let i = 0; i < bytes.length; i++) bin += String.fromCharCode(bytes[i]);
  return btoa(bin);
}

// ── Streaming Playback ──
function resetStreamPlaybackState() {
  currentSource = null;
  streamNextTime = 0;
  streamPlaybackStarted = false;
}

function syncStreamPlaybackState() {
  if (streamSources.length === 0) {
    resetStreamPlaybackState();
  }
}

function stopPlayback() {
  for (const src of streamSources) {
    try { src.stop(); } catch {}
  }
  streamSources = [];
  resetStreamPlaybackState();
}

function setListeningEnabled(enabled) {
  listeningEnabled = enabled;
  if (myvad) {
    try {
      if (enabled && !vadRunning) {
        if (typeof myvad.start === 'function') {
          myvad.start();
          vadRunning = true;
        }
      } else if (!enabled && vadRunning) {
        if (typeof myvad.pause === 'function') {
          myvad.pause();
          vadRunning = false;
        }
      }
    } catch (e) {
      console.warn('Failed to toggle VAD:', e.message);
    }
  }
  updateStateUI();
}

function toggleListening() {
  setListeningEnabled(!listeningEnabled);
}

// Status pill click handler — stop an in-flight generation when processing,
// otherwise toggle listening on/off.
if (statusEl) {
  statusEl.addEventListener('click', () => {
    if (state === 'processing') {
      requestInterrupt();
    } else {
      setListeningEnabled(!listeningEnabled);
    }
  });
}

function requestInterrupt() {
  if (state !== 'processing' && state !== 'speaking') return;
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: 'interrupt' }));
  }
  interruptedTurnId = activeTurnId;
  ignoreIncomingAudio = true;
  stopPlayback();
  setState('listening');
  console.log('User interrupted backend generation');
}

async function ensureAudioCtx() {
  if (!audioCtx) {
    audioCtx = new AudioContext();
    analyser = audioCtx.createAnalyser();
    analyser.fftSize = 256;
    analyser.smoothingTimeConstant = 0.75;
    try {
      audioStream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true } });
      getMicSource().connect(analyser);
    } catch (e) { console.warn('Mic capture failed:', e.message); }
  }
}

function startStreamPlayback() {
  stopPlayback();
  ensureAudioCtx();
  if (audioCtx.state === 'suspended') audioCtx.resume();
  streamNextTime = audioCtx.currentTime + 0.05;
}

function queueAudioChunk(base64Pcm) {
  ensureAudioCtx();

  const bin = atob(base64Pcm);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  const int16 = new Int16Array(bytes.buffer);
  const float32 = new Float32Array(int16.length);
  for (let i = 0; i < int16.length; i++) float32[i] = int16[i] / 32768;

  const audioBuffer = audioCtx.createBuffer(1, float32.length, streamSampleRate);
  audioBuffer.getChannelData(0).set(float32);

  const source = audioCtx.createBufferSource();
  source.buffer = audioBuffer;
  source.connect(audioCtx.destination);
  source.connect(analyser);

  const startAt = Math.max(streamNextTime, audioCtx.currentTime);
  if (!streamPlaybackStarted) {
    streamPlaybackStarted = true;
    speakingStartedAt = Date.now();
    setState('speaking');
  }
  source.start(startAt);
  streamNextTime = startAt + audioBuffer.duration;

  streamSources.push(source);
  currentSource = source;

  source.onended = () => {
    const idx = streamSources.indexOf(source);
    if (idx !== -1) streamSources.splice(idx, 1);
    if (currentSource === source) {
      currentSource = streamSources[streamSources.length - 1] || null;
    }
    syncStreamPlaybackState();
    if (streamSources.length === 0 && state === 'speaking') {
      setState('listening');
    }
  };
}

// ── UI ──
function addMessage(role, text, meta, isHtml = false) {
  const div = document.createElement('div');
  div.className = `msg ${role}`;
  if (isHtml) {
    div.innerHTML = text;
  } else {
    div.appendChild(document.createTextNode(text));
  }
  if (meta) {
    const metaDiv = document.createElement('div');
    metaDiv.className = 'meta';
    metaDiv.textContent = meta;
    div.appendChild(metaDiv);
  }
  messagesDiv.appendChild(div);
  return div;
}

function updateLatestUserMessage(text) {
  const userMsgs = messagesDiv.querySelectorAll('.msg.user');
  const lastUserMsg = userMsgs[userMsgs.length - 1];
  if (!lastUserMsg) return;

  const meta = lastUserMsg.querySelector('.meta');
  for (const node of [...lastUserMsg.childNodes]) {
    if (node !== meta) {
      lastUserMsg.removeChild(node);
    }
  }
  lastUserMsg.insertBefore(document.createTextNode(text), meta || null);
  scrollMessagesToBottom();
}

inputSourceSelect.addEventListener('change', () => {
  switchVideoSource(inputSourceSelect.value);
});

if (inputSourceSelectMobile) {
  inputSourceSelectMobile.addEventListener('change', () => {
    switchVideoSource(inputSourceSelectMobile.value);
  });
}

if (languageSelectMobile) {
  languageSelectMobile.addEventListener('change', () => {
    selectedLanguage = languageSelectMobile.value;
    renderVoiceOptions(selectedLanguage);
    saveTtsSelection(selectedLanguage, selectedVoice);
  });
}

if (voiceSelectMobile) {
  voiceSelectMobile.addEventListener('change', () => {
    selectedVoice = voiceSelectMobile.value;
    saveTtsSelection(selectedLanguage, selectedVoice);
  });
}

// ── Init ──
async function init() {
  initWaveformCanvas();
  window.addEventListener('resize', () => {
    refreshElements();
    scheduleWaveformCanvasResize();
  });
  document.addEventListener('visibilitychange', syncWakeLock);
  window.addEventListener('focus', syncWakeLock);
  window.addEventListener('pageshow', syncWakeLock);
  window.addEventListener('pagehide', releaseWakeLock);
  document.addEventListener('pointerdown', syncWakeLock, { passive: true });
  document.addEventListener('touchstart', syncWakeLock, { passive: true });

  ttsOptions = await fetch('/api/tts/options').then(response => response.json());
  const modelInfo = await fetch('/api/model').then(r => r.json()).catch(() => ({ model: null }));
  const modelLabel = document.getElementById('modelLabel');
  if (modelLabel) modelLabel.textContent = modelInfo.model || 'OpenAI-compatible backend';
  renderLanguageOptions();
  const saved = loadTtsSelection();
  if (saved) {
    setTtsSelection(saved.languageCode, saved.voice);
  } else {
    setTtsSelection(ttsOptions.default_language, ttsOptions.default_voice);
  }
  languageSelect.addEventListener('change', () => {
    selectedLanguage = languageSelect.value;
    renderVoiceOptions(selectedLanguage);
    saveTtsSelection(selectedLanguage, selectedVoice);
  });
  voiceSelect.addEventListener('change', () => {
    selectedVoice = voiceSelect.value;
    saveTtsSelection(selectedLanguage, selectedVoice);
  });

  await enumerateVideoDevices();
  if (preferredVideoSource !== 'off') {
    await switchVideoSource(preferredVideoSource, { persist: false, force: true });
  } else {
    applyVideoLayout();
  }
  connect();

  myvad = await vad.MicVAD.new({
    getStream: async () => {
      if (audioStream) { return audioStream; }
      try { return await navigator.mediaDevices.getUserMedia({ audio: true }); } catch { return new MediaStream(); }
    },
    positiveSpeechThreshold: 0.5,
    negativeSpeechThreshold: 0.25,
    redemptionMs: 600,
    minSpeechMs: 300,
    preSpeechPadMs: 300,
    onSpeechStart: handleSpeechStart,
    onSpeechEnd: handleSpeechEnd,
    onVADMisfire: () => { console.log('VAD misfire (too short)'); },
    onnxWASMBasePath: "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/",
    baseAssetPath: "https://cdn.jsdelivr.net/npm/@ricky0123/vad-web@0.0.29/dist/",
  });

  myvad.start();
  vadRunning = true;

  const initAudio = () => {
    ensureAudioCtx().then(() => {
      if (audioCtx.state === 'suspended') audioCtx.resume();
      syncWakeLock();
      document.removeEventListener('click', initAudio);
      document.removeEventListener('keydown', initAudio);
    });
  };
  document.addEventListener('click', initAudio);
  document.addEventListener('keydown', initAudio);
  ensureAudioCtx();

  setState('listening');
  setListeningEnabled(true);
  drawWaveform();

  console.log('VAD initialized and listening');
}

init();