/**
 * NOVA Kiosk UI — WebSocket-driven dashboard for the projector display.
 * Connects to the server, renders cards, handles confirmations + chat input.
 */

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

const API_BASE = `http://${window.location.hostname || 'localhost'}:8000`;
const WS_URL = `ws://${window.location.hostname || 'localhost'}:8000/ws/ui`;
const RECONNECT_INTERVAL_MS = 3000;
const TOAST_DEFAULT_DURATION = 4000;
const POLL_INTERVAL_MS = 15000; // Refresh devices/memories every 15s

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

let ws = null;
let reconnectTimer = null;
let pollTimer = null;
let cards = [];
let assistantState = 'idle';
let speakerVerified = false;
let speakerLabel = '';
let currentTranscript = '';
let micMuted = false;
let chatBusy = false;
let videoFeedActive = false;

// Cached data from REST API
let cachedDevices = [];
let cachedMemories = [];
let cachedConversations = [];

// ---------------------------------------------------------------------------
// DOM References
// ---------------------------------------------------------------------------

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const els = {
    connectionStatus: $('#connection-status'),
    stateDot: $('#state-dot'),
    stateLabel: $('#state-label'),
    transcript: $('#transcript-display'),
    speakerBadge: $('#speaker-badge'),
    clock: $('#clock'),
    assistantResponse: $('#assistant-response'),
    cardsArea: $('#cards-area'),
    conversationsList: $('#conversations-list'),
    memoryList: $('#memory-list'),
    devicesList: $('#devices-list'),
    deviceCount: $('#device-count'),
    pendingList: $('#pending-list'),
    toastContainer: $('#toast-container'),
    micToggle: $('#mic-toggle'),
    stopSpeaking: $('#stop-speaking'),
    chatInput: $('#chat-input'),
    chatForm: $('#chat-form'),
};

// ---------------------------------------------------------------------------
// WebSocket Connection
// ---------------------------------------------------------------------------

function connect() {
    if (ws && ws.readyState === WebSocket.OPEN) return;

    ws = new WebSocket(WS_URL);

    ws.onopen = () => {
        console.log('[WS] Connected');
        setConnectionStatus(true);
        clearTimeout(reconnectTimer);
        // Fetch initial data on connect
        fetchAllData();
    };

    ws.onclose = () => {
        console.log('[WS] Disconnected');
        setConnectionStatus(false);
        scheduleReconnect();
    };

    ws.onerror = (err) => {
        console.error('[WS] Error:', err);
    };

    ws.onmessage = (evt) => {
        try {
            const data = JSON.parse(evt.data);
            handleEvent(data);
        } catch (e) {
            console.error('[WS] Parse error:', e);
        }
    };
}

function scheduleReconnect() {
    clearTimeout(reconnectTimer);
    reconnectTimer = setTimeout(connect, RECONNECT_INTERVAL_MS);
}

function sendEvent(event) {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify(event));
    }
}

function setConnectionStatus(connected) {
    els.connectionStatus.textContent = connected ? 'Connected' : 'Disconnected';
    els.connectionStatus.className = connected ? 'badge badge-connected' : 'badge badge-disconnected';
}

// ---------------------------------------------------------------------------
// REST API Fetching
// ---------------------------------------------------------------------------

async function fetchJSON(path) {
    try {
        const resp = await fetch(`${API_BASE}${path}`);
        if (!resp.ok) return null;
        return await resp.json();
    } catch (e) {
        console.error(`[API] Failed to fetch ${path}:`, e);
        return null;
    }
}

async function fetchAllData() {
    const [devices, memories, conversations] = await Promise.all([
        fetchJSON('/api/devices'),
        fetchJSON('/api/memories'),
        fetchJSON('/api/conversations?limit=10'),
    ]);
    if (devices) { cachedDevices = devices; renderDevicesPanel(); }
    if (memories) { cachedMemories = memories; renderMemoriesPanel(); }
    if (conversations) { cachedConversations = conversations; renderConversationsPanel(); }
}

function startPolling() {
    pollTimer = setInterval(fetchAllData, POLL_INTERVAL_MS);
}

// ---------------------------------------------------------------------------
// Event Handling
// ---------------------------------------------------------------------------

function handleEvent(data) {
    const type = data.event;
    console.log('[Event]', type, data);

    switch (type) {
        case 'ui_status_update':
            updateStatus(data);
            break;
        case 'ui_cards_update':
            updateCards(data.cards || []);
            break;
        case 'ui_toast':
            showToast(data.message, data.level || 'info', data.duration_ms || TOAST_DEFAULT_DURATION);
            break;
        case 'assistant_text':
            showAssistantResponse(data.text);
            break;
        default:
            console.log('[Event] Unhandled:', type);
    }
}

// ---------------------------------------------------------------------------
// Status Updates
// ---------------------------------------------------------------------------

function updateStatus(data) {
    assistantState = data.assistant_state || 'idle';
    speakerVerified = data.speaker_verified || false;
    speakerLabel = data.speaker_label || '';
    currentTranscript = data.transcript || '';

    // State dot + label
    els.stateDot.className = `dot dot-${assistantState}`;
    els.stateLabel.textContent = capitalize(assistantState);

    // Transcript
    els.transcript.textContent = currentTranscript;

    // Stop button visibility
    updateStopButton();

    // Speaker badge
    if (speakerVerified) {
        els.speakerBadge.textContent = speakerLabel || 'Verified';
        els.speakerBadge.className = 'badge badge-verified';
    } else {
        els.speakerBadge.textContent = 'Not Verified';
        els.speakerBadge.className = 'badge badge-unverified';
    }
}

// ---------------------------------------------------------------------------
// Assistant Response
// ---------------------------------------------------------------------------

function showAssistantResponse(text) {
    if (!text) return;
    els.assistantResponse.innerHTML = `<div class="response-text">${escapeHtml(text)}</div>`;
}

// ---------------------------------------------------------------------------
// Chat Input
// ---------------------------------------------------------------------------

async function sendChatMessage(event) {
    event.preventDefault();
    const text = els.chatInput.value.trim();
    if (!text || chatBusy) return;

    chatBusy = true;
    els.chatInput.disabled = true;

    // Show user message immediately
    els.assistantResponse.innerHTML = `<div class="response-text" style="color: var(--text-muted);">You: ${escapeHtml(text)}</div>`;

    try {
        const resp = await fetch(`${API_BASE}/api/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text }),
        });
        const data = await resp.json();
        if (data.assistant) {
            showAssistantResponse(data.assistant);
        }
        // Refresh side panels after a chat
        setTimeout(fetchAllData, 500);
    } catch (e) {
        showToast('Failed to send message', 'error');
    } finally {
        chatBusy = false;
        els.chatInput.disabled = false;
        els.chatInput.value = '';
        els.chatInput.focus();
    }
}

// ---------------------------------------------------------------------------
// Side Panels (auto-fetched data)
// ---------------------------------------------------------------------------

function renderDevicesPanel() {
    els.deviceCount.textContent = cachedDevices.length;
    if (cachedDevices.length === 0) {
        els.devicesList.innerHTML = '<div class="empty-state">No devices registered</div>';
        return;
    }
    els.devicesList.innerHTML = cachedDevices.map(d => {
        const lastAction = d.state?.last_action || d.state?.power || '';
        const isOn = lastAction === 'on';
        const indicatorClass = lastAction ? (isOn ? 'on' : 'off') : 'unknown';
        return `
            <div class="device-row">
                <div class="device-indicator ${indicatorClass}"></div>
                <div class="device-info">
                    <div class="device-name">${escapeHtml(d.name)}</div>
                    <div class="device-meta">${escapeHtml(d.room || 'No room')} &middot; ${escapeHtml(d.protocol)}</div>
                </div>
                <span class="device-type-badge">${escapeHtml(d.device_type)}</span>
            </div>
        `;
    }).join('');
}

function renderMemoriesPanel() {
    if (cachedMemories.length === 0) {
        els.memoryList.innerHTML = '<div class="empty-state">No memories saved</div>';
        return;
    }
    els.memoryList.innerHTML = cachedMemories.slice(0, 15).map(m => `
        <div class="card card-memory">
            <div class="card-body">${escapeHtml(m.text)}</div>
            <div style="font-size:10px; color: var(--text-muted); margin-top: 4px;">${escapeHtml(m.created_at || '')}</div>
        </div>
    `).join('');
}

function renderConversationsPanel() {
    if (cachedConversations.length === 0) {
        els.conversationsList.innerHTML = '<div class="empty-state">No conversations yet</div>';
        return;
    }
    els.conversationsList.innerHTML = cachedConversations.slice(0, 10).map(s => {
        const intents = (s.intents || []).join(', ');
        return `
            <div class="convo-row">
                <span class="convo-date">${escapeHtml(s.started_at || '')}</span>
                <span class="convo-meta">${s.turn_count} turns &middot; ${escapeHtml(intents)}</span>
            </div>
        `;
    }).join('');
}

// ---------------------------------------------------------------------------
// Cards Rendering (WebSocket-pushed)
// ---------------------------------------------------------------------------

function updateCards(newCards) {
    cards = newCards;
    renderAllCards();
}

function renderAllCards() {
    const sorted = [...cards].sort((a, b) => (b.priority || 0) - (a.priority || 0));

    const memories = [];
    const devices = [];
    const pending = [];
    const center = [];

    for (const card of sorted) {
        switch (card.card_type) {
            case 'MemorySavedCard':
                memories.push(card);
                break;
            case 'DeviceStatusCard':
                devices.push(card);
                break;
            case 'PendingActionCard':
                pending.push(card);
                break;
            case 'AssistantResponseCard':
                showAssistantResponse(card.body);
                break;
            default:
                center.push(card);
                break;
        }
    }

    // Render pushed device cards (overlay on auto-fetched ones if present)
    if (devices.length > 0) {
        const deviceHtml = devices.map(renderDeviceCard).join('');
        els.devicesList.innerHTML = deviceHtml;
    }
    if (memories.length > 0) {
        const memHtml = memories.map(renderMemoryCard).join('');
        // Prepend pushed memories to fetched ones
        const existing = els.memoryList.innerHTML;
        els.memoryList.innerHTML = memHtml + existing;
    }
    els.pendingList.innerHTML = pending.length > 0
        ? pending.map(renderPendingCard).join('')
        : '<div class="empty-state">None</div>';
    els.cardsArea.innerHTML = center.map(renderCenterCard).join('');
}

// ---------------------------------------------------------------------------
// Card Renderers
// ---------------------------------------------------------------------------

function renderMemoryCard(card) {
    return `
        <div class="card card-memory">
            <div class="card-header">
                <span class="card-title">${escapeHtml(card.title)}</span>
            </div>
            <div class="card-body">${escapeHtml(typeof card.body === 'string' ? card.body : JSON.stringify(card.body))}</div>
        </div>
    `;
}

function renderDeviceCard(card) {
    const body = card.body;
    if (Array.isArray(body)) {
        const rows = body.map(d => {
            const lastAction = d.state?.last_action || d.state?.power || '';
            const isOn = lastAction === 'on';
            const indicatorClass = lastAction ? (isOn ? 'on' : 'off') : 'unknown';
            return `
                <div class="device-row">
                    <div class="device-indicator ${indicatorClass}"></div>
                    <div class="device-info">
                        <div class="device-name">${escapeHtml(d.name || d.id)}</div>
                        <div class="device-meta">${escapeHtml(d.room || '')} &middot; ${escapeHtml(d.device_type || d.type || '')}</div>
                    </div>
                    <span class="device-type-badge">${escapeHtml(d.device_type || d.type || '')}</span>
                </div>
            `;
        }).join('');
        return `
            <div class="card card-device">
                <div class="card-header"><span class="card-title">${escapeHtml(card.title)}</span></div>
                <div class="card-body">${rows}</div>
            </div>
        `;
    }
    return `
        <div class="card card-device">
            <div class="card-header"><span class="card-title">${escapeHtml(card.title)}</span></div>
            <div class="card-body"><pre>${escapeHtml(JSON.stringify(body, null, 2))}</pre></div>
        </div>
    `;
}

function renderPendingCard(card) {
    const body = card.body || {};
    const actionId = body.action_id || '';
    const actions = (card.actions || []).map(a => {
        if (a.action.startsWith('confirm_')) {
            return `<button class="btn btn-confirm" onclick="confirmAction('${actionId}', true)">CONFIRM</button>`;
        }
        return `<button class="btn btn-cancel" onclick="confirmAction('${actionId}', false)">CANCEL</button>`;
    }).join('');

    return `
        <div class="card card-pending">
            <div class="card-header">
                <span class="card-title">${escapeHtml(card.title)}</span>
            </div>
            <div class="card-body"><pre>${escapeHtml(body.preview || '')}</pre></div>
            <div class="card-actions">${actions}</div>
        </div>
    `;
}

function renderCenterCard(card) {
    const typeClass = getCardClass(card.card_type);
    const body = card.body;
    let bodyHtml = '';

    if (card.card_type === 'VisionCard') {
        // Vision cards may have base64 image
        if (typeof body === 'string') {
            bodyHtml = escapeHtml(body);
        } else if (body && body.image_base64) {
            bodyHtml = `<img class="vision-image" src="data:image/jpeg;base64,${body.image_base64}" alt="Camera capture" />`;
            if (body.description) {
                bodyHtml += `<p style="margin-top: 8px;">${escapeHtml(body.description)}</p>`;
            }
        } else {
            bodyHtml = `<pre>${escapeHtml(JSON.stringify(body, null, 2))}</pre>`;
        }
    } else if (card.card_type === 'EmailSummaryCard' && body && body.messages) {
        bodyHtml = body.messages.map(m => `
            <div class="email-row">
                <span class="email-from">${escapeHtml(m.from || '')}</span>
                <span class="email-subject">${escapeHtml(m.subject || '(no subject)')}</span>
                <span class="email-snippet">${escapeHtml(m.snippet || '')}</span>
            </div>
        `).join('');
    } else if (card.card_type === 'EmailDraftCard' || card.card_type === 'SlackDraftCard') {
        bodyHtml = `
            <div>
                <strong>To:</strong> ${escapeHtml(body?.to || body?.channel || '')}<br>
                <strong>Subject:</strong> ${escapeHtml(body?.subject || '')}<br>
                <pre>${escapeHtml(body?.body || body?.text || '')}</pre>
            </div>
        `;
    } else if (typeof body === 'string') {
        bodyHtml = escapeHtml(body);
    } else {
        bodyHtml = `<pre>${escapeHtml(JSON.stringify(body, null, 2))}</pre>`;
    }

    // Render action buttons
    const actions = (card.actions || []).map(a => {
        const isConfirm = a.label.toLowerCase().includes('confirm') || a.label.toLowerCase().includes('send') || a.label.toLowerCase().includes('post');
        const cls = isConfirm ? 'btn-confirm' : 'btn-cancel';
        return `<button class="btn ${cls}" onclick="handleCardAction('${escapeHtml(a.action)}', '${escapeHtml(card.card_id || '')}')">${escapeHtml(a.label)}</button>`;
    }).join('');

    return `
        <div class="card ${typeClass}">
            <div class="card-header">
                <span class="card-title">${escapeHtml(card.title)}</span>
                <span class="card-badge" style="background: var(--bg-primary); color: var(--text-muted);">${card.card_type.replace('Card', '')}</span>
            </div>
            <div class="card-body">${bodyHtml}</div>
            ${actions ? `<div class="card-actions">${actions}</div>` : ''}
        </div>
    `;
}

function getCardClass(cardType) {
    const map = {
        'EmailSummaryCard': 'card-email',
        'EmailDetailCard': 'card-email',
        'EmailDraftCard': 'card-email-draft',
        'EmailSentCard': 'card-sent',
        'SlackDraftCard': 'card-slack-draft',
        'SlackSentCard': 'card-sent',
        'MemorySavedCard': 'card-memory',
        'DeviceStatusCard': 'card-device',
        'PendingActionCard': 'card-pending',
        'AssistantResponseCard': 'card-response',
        'VisionCard': 'card-vision',
    };
    return map[cardType] || '';
}

// ---------------------------------------------------------------------------
// Actions
// ---------------------------------------------------------------------------

function confirmAction(actionId, confirmed) {
    sendEvent({
        event: 'user_confirmation',
        confirmed: confirmed,
        pending_action_id: actionId,
        timestamp: Date.now() / 1000,
    });
    showToast(confirmed ? 'Action confirmed.' : 'Action cancelled.', confirmed ? 'success' : 'warning');
}

function toggleMic() {
    micMuted = !micMuted;
    sendEvent({
        event: micMuted ? 'mic_mute' : 'mic_unmute',
        timestamp: Date.now() / 1000,
    });
    updateMicButton();
}

function updateMicButton() {
    if (!els.micToggle) return;
    if (micMuted) {
        els.micToggle.textContent = 'Start Listening';
        els.micToggle.classList.add('mic-muted');
    } else {
        els.micToggle.textContent = 'Stop Listening';
        els.micToggle.classList.remove('mic-muted');
    }
}

function stopSpeaking() {
    sendEvent({
        event: 'stop_speaking',
        timestamp: Date.now() / 1000,
    });
}

function updateStopButton() {
    const btn = $('#stop-speaking');
    if (!btn) return;
    btn.style.display = (assistantState === 'speaking') ? '' : 'none';
}

function handleCardAction(action, cardId) {
    if (action.startsWith('confirm_')) {
        confirmAction(cardId, true);
    } else if (action.startsWith('cancel_')) {
        confirmAction(cardId, false);
    }
}

// ---------------------------------------------------------------------------
// Toast
// ---------------------------------------------------------------------------

function showToast(message, level = 'info', duration = TOAST_DEFAULT_DURATION) {
    const toast = document.createElement('div');
    toast.className = `toast toast-${level}`;
    toast.textContent = message;
    toast.style.animationDuration = `${duration}ms`;
    els.toastContainer.appendChild(toast);
    setTimeout(() => toast.remove(), duration);
}

// ---------------------------------------------------------------------------
// Clock
// ---------------------------------------------------------------------------

function updateClock() {
    els.clock.textContent = new Date().toLocaleTimeString('en-US', {
        hour: '2-digit',
        minute: '2-digit',
        hour12: true,
    });
}

// ---------------------------------------------------------------------------
// Video Feed
// ---------------------------------------------------------------------------

function toggleVideoFeed() {
    const panel = document.getElementById('video-panel');
    const feed = document.getElementById('video-feed');
    const btn = document.getElementById('video-toggle-btn');

    if (videoFeedActive) {
        // Stop feed
        feed.src = '';
        panel.style.display = 'none';
        videoFeedActive = false;
    } else {
        // Start feed — MJPEG stream is simply an img src
        feed.src = `${API_BASE}/api/video/stream?fps=10`;
        panel.style.display = '';
        videoFeedActive = true;
        if (btn) btn.textContent = 'Stop Feed';
    }
}

// ---------------------------------------------------------------------------
// Keyboard shortcuts
// ---------------------------------------------------------------------------

document.addEventListener('keydown', (e) => {
    // Don't intercept when typing in chat input
    if (document.activeElement === els.chatInput) return;

    if (e.key === 'Escape') {
        if (assistantState === 'speaking') {
            stopSpeaking();
        } else if (videoFeedActive) {
            toggleVideoFeed();
        }
    } else if (e.key === 'f' || e.key === 'F11') {
        if (!document.fullscreenElement) {
            document.documentElement.requestFullscreen().catch(() => {});
        } else {
            document.exitFullscreen().catch(() => {});
        }
    } else if (e.key === 'v' || e.key === 'V') {
        // Toggle video feed
        toggleVideoFeed();
    } else if (e.key === '/' || e.key === 't' || e.key === 'T') {
        // Focus chat input
        e.preventDefault();
        els.chatInput.focus();
    }
});

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

function escapeHtml(str) {
    if (typeof str !== 'string') str = String(str || '');
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

function capitalize(str) {
    return str.charAt(0).toUpperCase() + str.slice(1);
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

function init() {
    connect();
    updateClock();
    setInterval(updateClock, 1000);
    startPolling();
    console.log('[NOVA] Kiosk UI initialized. Press F for fullscreen, T or / to type, V for video feed.');
}

init();
