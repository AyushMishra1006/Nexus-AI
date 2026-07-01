/* ═══════════════════════════════════════════════════════════════════════════
   Nexus AI — app.js v7 "Signal Deck"
   Sidebar on the left with 3D fold collapse · real three.js 3D nexus-core
   background (4 sources orbiting a synthesis core) · agent card 3D tilt ·
   template + suggestion-reveal interactions
   All SSE event handling preserved exactly from v6
   ═══════════════════════════════════════════════════════════════════════════ */


/* ══════════════════════════════════════════════════════════════════════════
   NEXUS CORE 3D — lightweight three.js background: a rotating wireframe
   core with the 4 sources orbiting it as colored nodes, connected by lines,
   plus a faint ambient particle field for depth. Deliberately low-poly/
   texture-free to stay light (no heavy scene/asset loading).
   ══════════════════════════════════════════════════════════════════════════ */

class NexusCore3D {
  constructor(canvas) {
    this.canvas = canvas;
    this._t0 = performance.now();
    this.mouse  = { x: 0, y: 0 };
    this.target = { x: 0, y: 0 };
    this._raf = null;

    this._resize      = this._resize.bind(this);
    this._pointerMove  = this._pointerMove.bind(this);
    this._animate      = this._animate.bind(this);

    this.scene  = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 100);
    this.camera.position.set(0, 0, 9);

    this.renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));

    this._buildScene();
    this._resize();

    window.addEventListener('resize', this._resize);
    window.addEventListener('pointermove', this._pointerMove);
    this._animate();
  }

  _resize() {
    const w = window.innerWidth, h = window.innerHeight;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  _pointerMove(e) {
    this.target.x = (e.clientX / window.innerWidth  - 0.5) * 2;
    this.target.y = (e.clientY / window.innerHeight - 0.5) * 2;
  }

  _buildScene() {
    // Synthesis core — wireframe icosahedron + soft inner glow shell
    const coreGeo = new THREE.IcosahedronGeometry(1.5, 1);
    const coreMat = new THREE.MeshBasicMaterial({ color: 0x16a34a, wireframe: true, transparent: true, opacity: 0.4 });
    this.core = new THREE.Mesh(coreGeo, coreMat);
    this.scene.add(this.core);

    const glowGeo = new THREE.IcosahedronGeometry(1.05, 1);
    const glowMat = new THREE.MeshBasicMaterial({ color: 0x16a34a, transparent: true, opacity: 0.06 });
    this.coreGlow = new THREE.Mesh(glowGeo, glowMat);
    this.scene.add(this.coreGlow);

    // The 4 sources, orbiting the core, each tied by a line
    const defs = [
      { color: 0x4df0ff, angle: 0 },              // Wikipedia — phosphor cyan
      { color: 0xffb020, angle: Math.PI / 2 },     // ArXiv — gold
      { color: 0xc65eff, angle: Math.PI },         // Web — violet
      { color: 0xff3b5c, angle: Math.PI * 1.5 },   // YouTube — crimson
    ];
    this.orbitRadius = 3.6;
    this.nodes = defs.map(d => {
      const mesh = new THREE.Mesh(
        new THREE.SphereGeometry(0.09, 12, 12),
        new THREE.MeshBasicMaterial({ color: d.color })
      );
      this.scene.add(mesh);

      const lineGeo = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0, 0, 0), new THREE.Vector3(0, 0, 0)]);
      const line = new THREE.Line(lineGeo, new THREE.LineBasicMaterial({ color: d.color, transparent: true, opacity: 0.32 }));
      this.scene.add(line);

      return { mesh, line, angle: d.angle, speed: 0.12 + Math.random() * 0.05, tilt: Math.random() * 0.6 - 0.3 };
    });

    // Ambient particle field
    const count = 260;
    const positions = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
      positions[i * 3]     = (Math.random() - 0.5) * 22;
      positions[i * 3 + 1] = (Math.random() - 0.5) * 22;
      positions[i * 3 + 2] = (Math.random() - 0.5) * 22 - 4;
    }
    const pGeo = new THREE.BufferGeometry();
    pGeo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    this.particles = new THREE.Points(pGeo, new THREE.PointsMaterial({
      color: 0x9c98a5, size: 0.02, transparent: true, opacity: 0.35, sizeAttenuation: true,
    }));
    this.scene.add(this.particles);
  }

  _animate() {
    const t = (performance.now() - this._t0) / 1000;

    this.core.rotation.x = t * 0.08;
    this.core.rotation.y = t * 0.12;
    this.coreGlow.rotation.copy(this.core.rotation);
    this.coreGlow.scale.setScalar(1 + Math.sin(t * 1.1) * 0.06);

    this.nodes.forEach(n => {
      const a = t * n.speed + n.angle;
      const x = Math.cos(a) * this.orbitRadius;
      const z = Math.sin(a) * this.orbitRadius;
      const y = Math.sin(a * 1.3 + n.tilt) * 0.8;
      n.mesh.position.set(x, y, z);
      const pos = n.line.geometry.attributes.position;
      pos.setXYZ(0, 0, 0, 0);
      pos.setXYZ(1, x, y, z);
      pos.needsUpdate = true;
    });

    this.particles.rotation.y = t * 0.01;

    this.mouse.x += (this.target.x - this.mouse.x) * 0.03;
    this.mouse.y += (this.target.y - this.mouse.y) * 0.03;
    this.camera.position.x = this.mouse.x * 1.2;
    this.camera.position.y = -this.mouse.y * 0.8;
    this.camera.lookAt(0, 0, 0);

    this.renderer.render(this.scene, this.camera);
    this._raf = requestAnimationFrame(this._animate);
  }

  destroy() {
    if (this._raf) cancelAnimationFrame(this._raf);
    window.removeEventListener('resize', this._resize);
    window.removeEventListener('pointermove', this._pointerMove);
  }
}


/* ══════════════════════════════════════════════════════════════════════════
   INTRO CONSTELLATION — a sparse graph of nodes behind the wordmark that
   continuously rewires itself: new links fade in, old ones fade out, so the
   wiring never holds still. Lives only for the duration of the intro.
   ══════════════════════════════════════════════════════════════════════════ */

class IntroConstellation {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.links = [];
    this.colors = ['#16a34a', '#4df0ff', '#ffb020', '#c65eff'];
    this._raf = null;
    this._resize = this._resize.bind(this);
    this._animate = this._animate.bind(this);

    window.addEventListener('resize', this._resize);
    this._resize();
    this._initNodes();
    this._rewireTimer = setInterval(() => this._addLink(), 420);
    this._animate();
  }

  _resize() {
    this.canvas.width = window.innerWidth;
    this.canvas.height = window.innerHeight;
  }

  _initNodes() {
    const cx = this.canvas.width / 2, cy = this.canvas.height / 2;
    const count = 16;
    this.nodes = Array.from({ length: count }, (_, i) => {
      const angle  = (i / count) * Math.PI * 2 + Math.random() * 0.4;
      const radius = 220 + Math.random() * 260;
      return {
        x: cx + Math.cos(angle) * radius,
        y: cy + Math.sin(angle) * radius,
        vx: (Math.random() - 0.5) * 0.15,
        vy: (Math.random() - 0.5) * 0.15,
        color: this.colors[i % this.colors.length],
      };
    });
  }

  _addLink() {
    if (this.links.length > 10) this.links.shift();
    const a = Math.floor(Math.random() * this.nodes.length);
    let b = Math.floor(Math.random() * this.nodes.length);
    while (b === a) b = Math.floor(Math.random() * this.nodes.length);
    this.links.push({ a, b, born: performance.now(), life: 3200 + Math.random() * 2000 });
  }

  _animate() {
    const { ctx, canvas, nodes, links } = this;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const now = performance.now();

    nodes.forEach(n => { n.x += n.vx; n.y += n.vy; });

    for (let i = links.length - 1; i >= 0; i--) {
      const l = links[i];
      const age = now - l.born;
      if (age > l.life) { links.splice(i, 1); continue; }
      const t = age / l.life;
      let alpha;
      if (t < 0.2) alpha = t / 0.2;
      else if (t > 0.7) alpha = 1 - (t - 0.7) / 0.3;
      else alpha = 1;

      const na = nodes[l.a], nb = nodes[l.b];
      ctx.save();
      ctx.globalAlpha = alpha * 0.5;
      ctx.strokeStyle = na.color;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(na.x, na.y);
      ctx.lineTo(nb.x, nb.y);
      ctx.stroke();
      ctx.restore();
    }

    nodes.forEach(n => {
      ctx.save();
      ctx.globalAlpha = 0.85;
      ctx.fillStyle = n.color;
      ctx.shadowColor = n.color;
      ctx.shadowBlur = 10;
      ctx.beginPath();
      ctx.arc(n.x, n.y, 2.6, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    });

    this._raf = requestAnimationFrame(this._animate);
  }

  destroy() {
    if (this._raf) cancelAnimationFrame(this._raf);
    if (this._rewireTimer) clearInterval(this._rewireTimer);
    window.removeEventListener('resize', this._resize);
  }
}


/* ══════════════════════════════════════════════════════════════════════════
   HELPERS
   ══════════════════════════════════════════════════════════════════════════ */

const delay = ms => new Promise(r => setTimeout(r, ms));

let _network = null;

function initBgCanvas() {
  const canvas = document.getElementById('bg-canvas');
  if (!canvas || _network) return;
  if (typeof THREE === 'undefined') return; // three.js CDN unavailable — skip background gracefully
  try {
    _network = new NexusCore3D(canvas);
  } catch (_) { /* WebGL unavailable on this device — app still works without the 3D background */ }
}


/* ══════════════════════════════════════════════════════════════════════════
   INTRO SEQUENCE — logo scales in, NEXUS letters assemble from the edges,
   AI locks in, tagline + source pills follow. Kept simple per feedback.
   ══════════════════════════════════════════════════════════════════════════ */

async function introSequence() {
  const overlay   = document.getElementById('intro-overlay');
  const app       = document.getElementById('app');
  const logo      = document.getElementById('intro-logo');
  const nexusWrap = document.getElementById('intro-nexus-wrap');
  const aiWrap    = document.getElementById('intro-ai-wrap');
  const tagline   = document.getElementById('intro-tagline');
  const sources   = document.getElementById('intro-sources');
  const boom      = document.getElementById('intro-boom');
  const center    = document.querySelector('.intro-center');

  // 3D background starts immediately (visible faintly through overlay)
  initBgCanvas();

  // Constellation rewires itself behind the wordmark for the intro's duration
  let constellation = null;
  const constellationCanvas = document.getElementById('intro-constellation');
  if (constellationCanvas) {
    try { constellation = new IntroConstellation(constellationCanvas); } catch (_) { /* canvas unavailable */ }
  }

  // Phase 1 — logo scales in
  await delay(200);
  logo.classList.add('show');

  // Phase 2 — letters of NEXUS assemble from the edges of the browser in 3D
  await delay(350);
  nexusWrap.classList.add('show');

  // Phase 3 — AI locks in once the letters have settled, with a volumetric
  // shockwave burst marking the moment the wordmark completes
  await delay(950);
  aiWrap.classList.add('show');
  boom.classList.add('fire');
  center.classList.add('boom');

  // Phase 4 — tagline fades in
  await delay(340);
  tagline.classList.add('show');

  // Phase 5 — source pills appear
  await delay(250);
  sources.classList.add('show');

  // Phase 6 — hold, then dissolve
  await delay(800);
  overlay.classList.add('fade-out');
  app.style.display = '';         // unhide app

  // Phase 7 — remove overlay from DOM after transition
  await delay(700);
  if (constellation) constellation.destroy();
  overlay.remove();
}


/* ══════════════════════════════════════════════════════════════════════════
   SIDEBAR TOGGLE
   ══════════════════════════════════════════════════════════════════════════ */

function toggleSidebar() {
  document.getElementById('app').classList.toggle('sidebar-collapsed');
}


/* ══════════════════════════════════════════════════════════════════════════
   SUGGESTIONS REVEAL — hidden behind a button, animates into a structured
   grid on click
   ══════════════════════════════════════════════════════════════════════════ */

function toggleSuggestions() {
  const grid    = document.getElementById('suggestion-grid');
  const chevron = document.getElementById('suggestions-chevron');
  const label   = document.getElementById('suggestions-toggle-label');
  const isHidden = grid.classList.contains('hidden');

  if (isHidden) {
    grid.classList.remove('hidden');
    void grid.offsetWidth; // restart the stagger animation each time it opens
    grid.classList.add('open');
    chevron.classList.add('open');
    label.textContent = 'Hide suggested questions';
  } else {
    grid.classList.add('hidden');
    grid.classList.remove('open');
    chevron.classList.remove('open');
    label.textContent = 'Show suggested questions';
  }
}


/* ══════════════════════════════════════════════════════════════════════════
   TEMPLATES — four starting modes; each primes the input with a matching hint
   ══════════════════════════════════════════════════════════════════════════ */

const TEMPLATE_HINTS = {
  deep:    'Ask for an exhaustive, multi-source deep dive on...',
  quick:   'Ask a quick, focused question you need a fast answer to...',
  compare: 'Ask how sources agree, differ, or contradict on...',
  trend:   'Ask about the newest research or trending discussion on...',
};

function selectTemplate(mode) {
  const hint = TEMPLATE_HINTS[mode];
  if (hint) inputEl.placeholder = hint;
  inputEl.focus();
}


/* ══════════════════════════════════════════════════════════════════════════
   TICKER — duplicate content for seamless CSS loop
   ══════════════════════════════════════════════════════════════════════════ */

function initTicker() {
  const el = document.getElementById('ticker-text');
  if (!el) return;
  const orig = el.innerHTML;
  el.innerHTML = orig + orig; // duplicate → animation loops at -50%
}


/* ══════════════════════════════════════════════════════════════════════════
   USER IDENTITY (localStorage UUID)
   ══════════════════════════════════════════════════════════════════════════ */

let userId = localStorage.getItem('nexus_uid');
if (!userId) {
  userId = (crypto.randomUUID
    ? crypto.randomUUID()
    : 'uid-' + Date.now() + '-' + Math.random().toString(36).slice(2));
  localStorage.setItem('nexus_uid', userId);
}


/* ══════════════════════════════════════════════════════════════════════════
   STATE
   ══════════════════════════════════════════════════════════════════════════ */

let isRunning  = false;
let rawAnswer  = '';
let currentEvt = null;


/* ══════════════════════════════════════════════════════════════════════════
   AGENT CONFIG
   ══════════════════════════════════════════════════════════════════════════ */

const AGENTS = {
  wikipedia: {
    fetch:   'Searching knowledge base...',
    done:    'Knowledge base ready',
    skipped: 'Not applicable',
    error:   'Unavailable',
    timeout: 'Timed out',
  },
  arxiv: {
    fetch:   'Scanning research papers...',
    done:    'Papers analyzed',
    skipped: 'No relevant papers',
    error:   'Unavailable',
    timeout: 'Timed out',
  },
  web: {
    fetch:   'Browsing live sources...',
    done:    'Web sources ready',
    skipped: 'No results found',
    error:   'Unavailable',
    timeout: 'Timed out',
  },
  youtube: {
    fetch:   'Finding explainer videos...',
    done:    'Videos processed',
    skipped: 'No relevant videos',
    error:   'Dropped',
    timeout: 'Timed out',
  },
};


/* ══════════════════════════════════════════════════════════════════════════
   DOM HELPERS
   ══════════════════════════════════════════════════════════════════════════ */

const $ = id => document.getElementById(id);


/* ══════════════════════════════════════════════════════════════════════════
   INPUT AUTO-RESIZE
   ══════════════════════════════════════════════════════════════════════════ */

const inputEl = $('query-input');
const runBtn  = $('run-btn');

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


/* ══════════════════════════════════════════════════════════════════════════
   HISTORY
   ══════════════════════════════════════════════════════════════════════════ */

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
    div.className     = 'history-item';
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
  document.querySelectorAll('.history-item').forEach(d => d.classList.remove('active'));
  divEl.classList.add('active');

  $('welcome').classList.add('hidden');
  const ra = $('response-area');
  ra.classList.remove('hidden');

  $('query-echo').innerHTML = `<span class="query-label">Research Query</span>${escapeHtml(item.query_text || '')}`;
  $('agents-grid').classList.add('hidden');
  $('status-line').classList.add('hidden');
  $('error-banner').classList.add('hidden');

  const ac   = $('answer-card');
  const body = $('answer-body');
  ac.classList.remove('hidden');
  body.classList.remove('streaming');
  body.innerHTML = item.answer_text
    ? marked.parse(item.answer_text)
    : '<em style="color:var(--text-muted)">No answer saved.</em>';

  renderMeta(item.total_time, item.quality);

  $('sources-panel').classList.add('hidden');
  $('sources-toggle').classList.remove('hidden');
  $('sources-panel').innerHTML = '';
  const sourcesUsed = Array.isArray(item.sources_used) ? item.sources_used : [];
  if (sourcesUsed.length > 0) renderSourceCards(sourcesUsed);

  $('scroll-area').scrollTop = 0;
}


/* ══════════════════════════════════════════════════════════════════════════
   MAIN QUERY FLOW
   ══════════════════════════════════════════════════════════════════════════ */

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

  isRunning  = true;
  runBtn.disabled = true;
  rawAnswer  = '';

  $('welcome').classList.add('hidden');
  const ra = $('response-area');
  ra.classList.remove('hidden');

  $('query-echo').innerHTML = `<span class="query-label">Research Query</span>${escapeHtml(query)}`;

  Object.keys(AGENTS).forEach(src => resetAgentCard(src));
  $('agents-grid').classList.remove('hidden');

  $('status-line').classList.remove('hidden');
  $('status-pulse').className = 'status-pulse';
  setStatus('Understanding your question...');

  const ac = $('answer-card');
  ac.classList.add('hidden');
  $('answer-body').innerHTML = '';
  $('answer-body').classList.remove('streaming');
  $('sources-panel').innerHTML = '';
  $('sources-panel').classList.add('hidden');
  $('sources-toggle').classList.remove('hidden');
  $('answer-meta').innerHTML = '';
  $('error-banner').classList.add('hidden');

  $('scroll-area').scrollTop = 0;

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


/* ══════════════════════════════════════════════════════════════════════════
   SSE EVENT HANDLER  (identical to v2 — no changes to event logic)
   ══════════════════════════════════════════════════════════════════════════ */

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
      const src    = ev.source;
      const status = ev.status;
      if (status === 'TITLE_SELECTED') {
        setAgentCard(src, 'fetching', AGENTS[src].fetch);
      } else if (status === 'SUCCESS' || status === 'PARTIAL') {
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
      if (Array.isArray(ev.sources) && ev.sources.length > 0) {
        renderSourceCards(ev.sources);
      }
      break;

    case 'done': {
      Object.keys(AGENTS).forEach(src => {
        const card = $(`agent-${src}`);
        if (card && card.classList.contains('fetching')) {
          setAgentCard(src, 'timeout', AGENTS[src].timeout);
        }
      });

      const body = $('answer-body');
      body.classList.remove('streaming');
      if (rawAnswer) body.innerHTML = marked.parse(rawAnswer);

      renderMeta(ev.total_time, ev.quality);
      setStatus('Done', true);

      isRunning = false;
      runBtn.disabled = !inputEl.value.trim();

      loadHistory();

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
  if (!isRunning) return;
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


/* ══════════════════════════════════════════════════════════════════════════
   AGENT CARD 3D TILT — subtle pointer-driven perspective tilt
   ══════════════════════════════════════════════════════════════════════════ */

function attachCardTilt() {
  document.querySelectorAll('.agent-card, .template-card').forEach(card => {
    card.addEventListener('pointermove', e => {
      const r  = card.getBoundingClientRect();
      const px = (e.clientX - r.left) / r.width;
      const py = (e.clientY - r.top) / r.height;
      const rx = (0.5 - py) * 14;
      const ry = (px - 0.5) * 14;
      card.style.transform = `perspective(700px) rotateX(${rx}deg) rotateY(${ry}deg) translateZ(6px)`;
    });
    card.addEventListener('pointerleave', () => {
      card.style.transform = '';
    });
  });
}


/* ══════════════════════════════════════════════════════════════════════════
   AGENT CARD HELPERS
   ══════════════════════════════════════════════════════════════════════════ */

function resetAgentCard(src) {
  const card   = $(`agent-${src}`);
  const status = $(`agent-${src}-status`);
  const detail = $(`agent-${src}-detail`);
  card.className      = 'agent-card';
  status.textContent  = 'Waiting';
  detail.textContent  = '';
  // Reset progress bar
  const bar = card.querySelector('.agent-bar');
  if (bar) bar.style.width = '0';
}

function setAgentCard(src, state, statusText, detailText) {
  const card   = $(`agent-${src}`);
  const status = $(`agent-${src}-status`);
  const detail = $(`agent-${src}-detail`);
  card.className      = `agent-card ${state}`;
  status.textContent  = statusText  || '';
  detail.textContent  = detailText  || '';
}


/* ══════════════════════════════════════════════════════════════════════════
   STATUS LINE
   ══════════════════════════════════════════════════════════════════════════ */

function setStatus(text, done) {
  $('status-text').textContent = text;
  if (done) {
    $('status-pulse').classList.add('done');
  } else {
    $('status-pulse').classList.remove('done');
  }
}


/* ══════════════════════════════════════════════════════════════════════════
   SOURCE CARDS
   ══════════════════════════════════════════════════════════════════════════ */

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


/* ══════════════════════════════════════════════════════════════════════════
   META BAR
   ══════════════════════════════════════════════════════════════════════════ */

function renderMeta(totalTime, quality) {
  const meta = $('answer-meta');
  meta.innerHTML = '';

  if (totalTime) {
    const b = document.createElement('div');
    b.className   = 'meta-badge';
    b.textContent = `${totalTime}s`;
    meta.appendChild(b);
  }

  if (quality) {
    const q   = typeof quality === 'object' ? (quality.label || JSON.stringify(quality)) : quality;
    const cls = q === 'GOOD' ? 'good' : q === 'PASS' ? 'pass' : 'fail';
    const b   = document.createElement('div');
    b.className   = `meta-badge ${cls}`;
    b.textContent = q;
    meta.appendChild(b);
  }
}


/* ══════════════════════════════════════════════════════════════════════════
   ERROR
   ══════════════════════════════════════════════════════════════════════════ */

function showError(msg) {
  const banner      = $('error-banner');
  banner.textContent = msg;
  banner.classList.remove('hidden');
}


/* ══════════════════════════════════════════════════════════════════════════
   UTILITIES
   ══════════════════════════════════════════════════════════════════════════ */

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}


/* ══════════════════════════════════════════════════════════════════════════
   INIT
   ══════════════════════════════════════════════════════════════════════════ */

document.addEventListener('DOMContentLoaded', () => {
  // Hide app until intro completes
  const app = document.getElementById('app');
  app.style.display = 'none';

  // Duplicate ticker text for seamless loop
  initTicker();

  // Enable 3D tilt on agent cards
  attachCardTilt();

  // Run intro then load history
  introSequence().then(() => {
    loadHistory();
    inputEl.focus();
  });
});
