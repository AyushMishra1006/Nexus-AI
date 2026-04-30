/* ═══════════════════════════════════════════════════════════════════════════
   Nexus AI — app.js
   ═══════════════════════════════════════════════════════════════════════════ */

/* ── User identity (localStorage UUID) ──────────────────────────────────── */
let userId = localStorage.getItem('nexus_uid');
if (!userId) {
  userId = (crypto.randomUUID ? crypto.randomUUID() : 'uid-' + Date.now() + '-' + Math.random().toString(36).slice(2));
  localStorage.setItem('nexus_uid', userId);
}

/* ── State ───────────────────────────────────────────────────────────────── */
let isRunning  = false;
let rawAnswer  = '';        // accumulated raw tokens
let currentEvt = null;      // active EventSource

/* ── Agent config ────────────────────────────────────────────────────────── */
const AGENTS = {
  wikipedia: {
    fetch:    'Searching knowledge base...',
    done:     'Knowledge base ready',
    skipped:  'Not applicable',
    error:    'Unavailable',
    timeout:  'Timed out',
  },
  arxiv: {
    fetch:    'Scanning research papers...',
    done:     'Papers analyzed',
    skipped:  'No relevant papers',
    error:    'Unavailable',
    timeout:  'Timed out',
  },
  web: {
    fetch:    'Browsing the web...',
    done:     'Web sources ready',
    skipped:  'No web results',
    error:    'Unavailable',
    timeout:  'Timed out',
  },
  youtube: {
    fetch:    'Finding explainer videos...',
    done:     'Videos processed',
    skipped:  'No relevant videos',
    error:    'Unavailable',
    timeout:  'Timed out',
  },
};

/* ── DOM helpers ─────────────────────────────────────────────────────────── */
const $  = id => document.getElementById(id);
const el = id => $(id);

/* ── Input auto-resize ───────────────────────────────────────────────────── */
const inputEl  = $('query-input');
const runBtn   = $('run-btn');

inputEl.addEventListener('input', () => {
  inputEl.style.height = 'auto';
  inputEl.style.height = Math.min(inputEl.scrollHeight, 160) + 'px';
  runBtn.disabled = !inputEl.value.trim() || isRunning;
});

inputEl.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    if (!isRunning && inputEl.value.trim()) runQuery();
  }
});

/* ── Load history on page open ───────────────────────────────────────────── */
async function loadHistory() {
  try {
    const resp = await fetch(`/api/history/${userId}`);
    if (!resp.ok) return;
    const items = await resp.json();
    if (!Array.isArray(items) || items.length === 0) return;
    renderHistorySidebar(items);
  } catch (_) { /* Supabase not configured — silently skip */ }
}

function renderHistorySidebar(items) {
  const list  = $('history-list');
  const empty = $('history-empty');
  if (empty) empty.remove();

  list.innerHTML = '';
  items.forEach(item => {
    const div = document.createElement('div');
    div.className = 'history-item';
    div.dataset.queryId = item.query_id;

    const timeStr = item.timestamp
      ? new Date(item.timestamp).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
      : '';

    div.innerHTML = `
      <div class="history-item-text">${escapeHtml(item.query_text || '')}</div>
      <div class="history-item-time">${timeStr}</div>
    `;
    div.addEventListener('click', () => showHistoryItem(item, div));
    list.appendChild(div);
  });
}

function showHistoryItem(item, divEl) {
  // Mark active
  document.querySelectorAll('.history-item').forEach(d => d.classList.remove('active'));
  divEl.classList.add('active');

  // Hide welcome, show response area
  $('welcome').classList.add('hidden');
  const ra = $('response-area');
  ra.classList.remove('hidden');

  // Fill query echo
  $('query-echo').innerHTML = `<span class="query-label">Research Query</span>${escapeHtml(item.query_text || '')}`;

  // Hide agents + status (this is a history view, not a live run)
  $('agents-grid').classList.add('hidden');
  $('status-line').classList.add('hidden');
  $('error-banner').classList.add('hidden');

  // Show answer
  const ac = $('answer-card');
  ac.classList.remove('hidden');
  const body = $('answer-body');
  body.classList.remove('streaming');
  body.innerHTML = item.answer_text
    ? marked.parse(item.answer_text)
    : '<em style="color:var(--text-muted)">No answer saved.</em>';

  // Meta
  renderMeta(item.total_time, item.quality);

  // Sources (from sources_used if available)
  $('sources-panel').classList.add('hidden');
  $('sources-toggle').classList.remove('hidden');
  $('sources-panel').innerHTML = '';
  const sourcesUsed = Array.isArray(item.sources_used) ? item.sources_used : [];
  if (sourcesUsed.length > 0) renderSourceCards(sourcesUsed);

  // Scroll to top
  $('scroll-area').scrollTop = 0;
}

/* ── Main query flow ─────────────────────────────────────────────────────── */
function setQuery(text) {
  inputEl.value = text;
  inputEl.style.height = 'auto';
  inputEl.style.height = Math.min(inputEl.scrollHeight, 160) + 'px';
  runBtn.disabled = false;
  inputEl.focus();
}

function newQuery() {
  if (isRunning && currentEvt) { currentEvt.close(); currentEvt = null; }
  isRunning = false;
  inputEl.value = '';
  inputEl.style.height = 'auto';
  runBtn.disabled = true;

  $('welcome').classList.remove('hidden');
  $('response-area').classList.add('hidden');
  document.querySelectorAll('.history-item').forEach(d => d.classList.remove('active'));
  inputEl.focus();
}

function runQuery() {
  const query = inputEl.value.trim();
  if (!query || isRunning) return;

  isRunning = true;
  runBtn.disabled = true;
  rawAnswer = '';

  // Switch to response view
  $('welcome').classList.add('hidden');
  const ra = $('response-area');
  ra.classList.remove('hidden');

  // Query echo
  $('query-echo').innerHTML = `<span class="query-label">Research Query</span>${escapeHtml(query)}`;

  // Reset agent cards
  Object.keys(AGENTS).forEach(src => resetAgentCard(src));
  $('agents-grid').classList.remove('hidden');

  // Reset status line
  $('status-line').classList.remove('hidden');
  $('status-pulse').className = 'status-pulse';
  setStatus('Understanding your question...');

  // Hide/reset answer card
  const ac = $('answer-card');
  ac.classList.add('hidden');
  $('answer-body').innerHTML = '';
  $('answer-body').classList.remove('streaming');
  $('sources-panel').innerHTML = '';
  $('sources-panel').classList.add('hidden');
  $('sources-toggle').classList.remove('hidden');
  $('answer-meta').innerHTML = '';
  $('error-banner').classList.add('hidden');

  // Scroll to top
  $('scroll-area').scrollTop = 0;

  // Open SSE stream (async/await — same approach as original app)
  _streamQuery(query);
}

async function _streamQuery(query) {
  try {
    const resp = await fetch('/api/query', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ query, user_id: userId }),
    });

    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      throw new Error(err.error || `Server error ${resp.status}`);
    }

    const reader  = resp.body.getReader();
    const decoder = new TextDecoder();
    let   buffer  = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop();
      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const raw = line.slice(6).trim();
          if (raw) try { handleEvent(JSON.parse(raw)); } catch (_) {}
        }
      }
    }
  } catch (err) {
    showError(err.message || 'Connection failed.');
  } finally {
    onStreamEnd();
  }
}

/* ── SSE event handler ───────────────────────────────────────────────────── */
function handleEvent(ev) {
  switch (ev.type) {

    case 'start':
      setStatus('Gathering information from 4 sources...');
      break;

    case 'step_done':
      if (ev.step === 'rewrite') {
        setStatus('Gathering information from 4 sources...');
      } else if (ev.step === 'retrieve') {
        setStatus('Selecting the best information...');
      }
      break;

    case 'source_arrived': {
      const src = ev.source;
      const status = ev.status;
      if (status === 'TITLE_SELECTED') {
        setAgentCard(src, 'fetching', AGENTS[src].fetch);
      } else if (status === 'SUCCESS' || status === 'PARTIAL') {
        // Keep fetching state — will update to done after source_processed
        setAgentCard(src, 'fetching', AGENTS[src].fetch);
      } else if (status === 'SKIPPED') {
        setAgentCard(src, 'skipped', AGENTS[src].skipped);
      } else if (status === 'TIMEOUT') {
        setAgentCard(src, 'timeout', AGENTS[src].timeout);
      } else if (status === 'ERROR' || status === 'FAILED' || status === 'UNKNOWN') {
        setAgentCard(src, 'error', AGENTS[src].error);
      } else {
        setAgentCard(src, 'fetching', AGENTS[src].fetch);
      }
      break;
    }

    case 'source_processed': {
      const src = ev.source;
      const n   = ev.chunks || 0;
      if (n > 0) {
        setAgentCard(src, 'done', AGENTS[src].done, `${n} chunks`);
      } else {
        // Check if it was already set to skipped/error — don't overwrite
        const card = $(`agent-${src}`);
        if (!card.classList.contains('skipped') && !card.classList.contains('error')) {
          setAgentCard(src, 'skipped', AGENTS[src].skipped);
        }
      }
      break;
    }

    case 'step_start':
      if (ev.step === 'llm') {
        setStatus('Synthesizing your answer...');
        // Show answer card with streaming cursor
        const ac = $('answer-card');
        ac.classList.remove('hidden');
        $('answer-body').classList.add('streaming');
      } else if (ev.step === 'retrieve') {
        setStatus('Selecting the best information...');
      }
      break;

    case 'llm_token':
      if (ev.token) {
        rawAnswer += ev.token;
        $('answer-body').textContent = rawAnswer;
      }
      break;

    case 'llm_retry':
      setStatus(`Rate limited — retrying in ${ev.wait}s...`);
      break;

    case 'answer':
      // Sources arrive here
      if (Array.isArray(ev.sources) && ev.sources.length > 0) {
        renderSourceCards(ev.sources);
      }
      break;

    case 'done': {
      // Sweep any cards still blinking — timed out without receiving a status event
      Object.keys(AGENTS).forEach(src => {
        const card = $(`agent-${src}`);
        if (card && card.classList.contains('fetching')) {
          setAgentCard(src, 'timeout', AGENTS[src].timeout);
        }
      });

      // Render markdown
      const body = $('answer-body');
      body.classList.remove('streaming');
      if (rawAnswer) body.innerHTML = marked.parse(rawAnswer);

      // Meta
      renderMeta(ev.total_time, ev.quality);

      setStatus('Done', true);

      // Re-enable input
      isRunning = false;
      runBtn.disabled = !inputEl.value.trim();

      // Reload history sidebar
      loadHistory();

      // Scroll answer into view
      setTimeout(() => {
        $('answer-card').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      }, 100);
      break;
    }

    case 'error':
      showError(ev.message || 'An error occurred.');
      $('answer-body').classList.remove('streaming');
      setStatus('Something went wrong.', true);
      isRunning = false;
      runBtn.disabled = !inputEl.value.trim();
      break;

    case 'stream_end':
      onStreamEnd();
      break;
  }
}

function onStreamEnd() {
  if (isRunning) {
    // Safety sweep — stop any card still blinking when stream closes
    Object.keys(AGENTS).forEach(src => {
      const card = $(`agent-${src}`);
      if (card && card.classList.contains('fetching')) {
        setAgentCard(src, 'timeout', AGENTS[src].timeout);
      }
    });
    isRunning = false;
    runBtn.disabled = !inputEl.value.trim();
    $('answer-body').classList.remove('streaming');
    if (rawAnswer && !$('answer-card').classList.contains('hidden')) {
      $('answer-body').innerHTML = marked.parse(rawAnswer);
    }
  }
}

/* ── Agent card helpers ──────────────────────────────────────────────────── */
function resetAgentCard(src) {
  const card   = $(`agent-${src}`);
  const status = $(`agent-${src}-status`);
  const detail = $(`agent-${src}-detail`);
  const dot    = $(`agent-${src}-dot`);
  card.className   = 'agent-card';
  status.textContent = 'Waiting...';
  detail.textContent = '';
}

function setAgentCard(src, state, statusText, detailText) {
  const card   = $(`agent-${src}`);
  const status = $(`agent-${src}-status`);
  const detail = $(`agent-${src}-detail`);
  card.className       = `agent-card ${state}`;
  status.textContent   = statusText || '';
  detail.textContent   = detailText || '';
}

/* ── Status line ─────────────────────────────────────────────────────────── */
function setStatus(text, done) {
  $('status-text').textContent = text;
  if (done) {
    $('status-pulse').classList.add('done');
  } else {
    $('status-pulse').classList.remove('done');
  }
}

/* ── Sources ─────────────────────────────────────────────────────────────── */
function renderSourceCards(sources) {
  const panel = $('sources-panel');
  panel.innerHTML = '';
  const seen = new Set();
  sources.forEach(s => {
    const url = s.url || '';
    if (!url || seen.has(url)) return;
    seen.add(url);

    const src   = (s.source || 'web').toLowerCase();
    const title = s.title || url;
    const a     = document.createElement('a');
    a.className = 'source-card';
    a.href      = url;
    a.target    = '_blank';
    a.rel       = 'noopener noreferrer';
    a.innerHTML = `
      <div class="source-card-label ${src}">${src.toUpperCase()}</div>
      <div class="source-card-title">${escapeHtml(title)}</div>
    `;
    panel.appendChild(a);
  });
}

function toggleSources() {
  const panel   = $('sources-panel');
  const chevron = $('sources-chevron');
  const isHidden = panel.classList.contains('hidden');
  panel.classList.toggle('hidden', !isHidden);
  chevron.classList.toggle('open', isHidden);
}

/* ── Meta bar ────────────────────────────────────────────────────────────── */
function renderMeta(totalTime, quality) {
  const meta = $('answer-meta');
  meta.innerHTML = '';

  if (totalTime) {
    const b = document.createElement('div');
    b.className = 'meta-badge';
    b.textContent = `${totalTime}s`;
    meta.appendChild(b);
  }

  if (quality) {
    const q = typeof quality === 'object' ? (quality.label || JSON.stringify(quality)) : quality;
    const cls = q === 'GOOD' ? 'good' : q === 'PASS' ? 'pass' : 'fail';
    const b = document.createElement('div');
    b.className = `meta-badge ${cls}`;
    b.textContent = q;
    meta.appendChild(b);
  }
}

/* ── Error ───────────────────────────────────────────────────────────────── */
function showError(msg) {
  const banner = $('error-banner');
  banner.textContent = msg;
  banner.classList.remove('hidden');
}

/* ── Utilities ───────────────────────────────────────────────────────────── */
function escapeHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}


/* ── Init ────────────────────────────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => {
  loadHistory();
  inputEl.focus();
});
