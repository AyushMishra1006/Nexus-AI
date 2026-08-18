// Buddy — frontend. Vanilla JS, no build step, no CDN dependencies.

const state = {
  passcode: localStorage.getItem('companion_passcode') || '',
  config: { her_name: '', buddy_name: 'Buddy' },
  chatLoaded: false,
  moodSelected: null,
  profileText: '',
};

const $ = (sel) => document.querySelector(sel);
const $all = (sel) => document.querySelectorAll(sel);

function authHeaders(extra = {}) {
  const headers = { 'Content-Type': 'application/json', ...extra };
  if (state.passcode) headers['X-Companion-Passcode'] = state.passcode;
  return headers;
}

async function api(path, options = {}) {
  const resp = await fetch(path, {
    ...options,
    headers: authHeaders(options.headers || {}),
  });
  if (!resp.ok) {
    const err = new Error(`Request failed: ${resp.status}`);
    err.status = resp.status;
    throw err;
  }
  return resp.json();
}

// ── Floating decor ─────────────────────────────────────────────────────────

function spawnFloaties() {
  const container = $('#floaties');
  const emojis = ['💗', '✨', '🌸', '💌', '🎀'];
  for (let i = 0; i < 14; i++) {
    const span = document.createElement('span');
    span.className = 'floaty';
    span.textContent = emojis[Math.floor(Math.random() * emojis.length)];
    span.style.left = `${Math.random() * 100}%`;
    span.style.animationDuration = `${14 + Math.random() * 12}s`;
    span.style.animationDelay = `${Math.random() * 14}s`;
    span.style.fontSize = `${1 + Math.random() * 1.2}rem`;
    container.appendChild(span);
  }
}

// ── Boot sequence ─────────────────────────────────────────────────────────

async function boot() {
  spawnFloaties();

  let locked = false;
  try {
    const status = await fetch('/api/lock-status').then((r) => r.json());
    locked = status.locked;
  } catch (_) { /* backend not reachable yet — ignore */ }

  if (locked) {
    const ok = state.passcode && await checkPasscode(state.passcode);
    if (!ok) {
      showScreen('lock-screen');
      wireLockForm();
      return;
    }
  }

  await afterUnlock();
}

async function checkPasscode(passcode) {
  try {
    const resp = await fetch('/api/unlock', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ passcode }),
    });
    const data = await resp.json();
    return !!data.ok;
  } catch (_) {
    return false;
  }
}

function wireLockForm() {
  $('#lock-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const val = $('#lock-input').value.trim();
    const ok = await checkPasscode(val);
    if (!ok) {
      $('#lock-error').classList.remove('hidden');
      return;
    }
    state.passcode = val;
    localStorage.setItem('companion_passcode', val);
    await afterUnlock();
  });
}

async function afterUnlock() {
  const onboarding = await api('/api/onboarding');
  if (!onboarding.onboarded) {
    showScreen('onboarding-screen');
    wireOnboardingForm();
    return;
  }
  state.config = onboarding;
  showScreen('app');
  initApp();
}

function wireOnboardingForm() {
  $('#onboarding-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const her_name = $('#her-name-input').value.trim();
    const buddy_name = $('#buddy-name-input').value.trim() || 'Buddy';
    if (!her_name) return;
    const config = await api('/api/onboarding', {
      method: 'POST',
      body: JSON.stringify({ her_name, buddy_name }),
    });
    state.config = config;
    showScreen('app');
    initApp();
  });
}

function showScreen(id) {
  $all('.screen').forEach((el) => el.classList.add('hidden'));
  $(`#${id}`).classList.remove('hidden');
}

// ── App init ────────────────────────────────────────────────────────────────

function initApp() {
  $('#buddy-name-label').textContent = state.config.buddy_name;
  $('#home-greeting').textContent = `hi ${state.config.her_name} 🌷`;

  wireTabs();
  wireChat();
  wireTasks();
  wireCheckin();
  wireMemory();

  loadHome();
}

function wireTabs() {
  const goTo = (tab) => {
    $all('.tab-panel').forEach((p) => p.classList.toggle('active', p.dataset.tab === tab));
    $all('.tab-btn').forEach((b) => b.classList.toggle('active', b.dataset.tab === tab));
    if (tab === 'chat' && !state.chatLoaded) loadChatHistory();
    if (tab === 'tasks') loadTasks();
    if (tab === 'checkin') loadMood();
    if (tab === 'memory') loadMemory();
  };
  $all('.tab-btn').forEach((btn) => btn.addEventListener('click', () => goTo(btn.dataset.tab)));
  $all('[data-goto]').forEach((btn) => btn.addEventListener('click', () => goTo(btn.dataset.goto)));
  goTo('home');
}

// ── Home ───────────────────────────────────────────────────────────────────

async function loadHome() {
  api('/api/affirmation').then((res) => {
    $('#affirmation-text').textContent = res.text;
  }).catch(() => {
    $('#affirmation-text').textContent = "Buddy's still waking up — try again in a moment 💤";
  });

  api('/api/tasks').then((tasks) => {
    const open = tasks.filter((t) => !t.done).length;
    $('#open-tasks-count').textContent = open === 1 ? '1 task' : `${open} tasks`;
  }).catch(() => {});

  api('/api/mood').then((res) => {
    if (res.streak > 0) {
      $('#streak-badge').classList.remove('hidden');
      $('#streak-count').textContent = res.streak;
    }
  }).catch(() => {});
}

// ── Chat ───────────────────────────────────────────────────────────────────

function addBubble(text, who, streaming = false) {
  const div = document.createElement('div');
  div.className = `bubble ${who}${streaming ? ' streaming' : ''}`;
  div.textContent = text;
  $('#chat-log').appendChild(div);
  $('#chat-log').scrollTop = $('#chat-log').scrollHeight;
  return div;
}

async function loadChatHistory() {
  state.chatLoaded = true;
  try {
    const history = await api('/api/chat/history');
    $('#chat-log').innerHTML = '';
    if (!history.length) {
      addBubble(`Hi ${state.config.her_name}! What's on your mind today? 💗`, 'buddy');
      return;
    }
    history.forEach((turn) => addBubble(turn.text, turn.role === 'user' ? 'her' : 'buddy'));
  } catch (_) {
    addBubble("Couldn't load our chat history — but I'm still here!", 'buddy');
  }
}

function wireChat() {
  $('#chat-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const input = $('#chat-input');
    const message = input.value.trim();
    if (!message) return;
    input.value = '';
    input.disabled = true;

    addBubble(message, 'her');
    const buddyBubble = addBubble('', 'buddy', true);

    try {
      const resp = await fetch('/api/chat', {
        method: 'POST',
        headers: authHeaders(),
        body: JSON.stringify({ message }),
      });
      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let fullText = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop();
        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          const raw = line.slice(6).trim();
          if (!raw) continue;
          let ev;
          try { ev = JSON.parse(raw); } catch (_) { continue; }

          if (ev.type === 'token') {
            fullText += ev.text;
            buddyBubble.textContent = fullText;
            $('#chat-log').scrollTop = $('#chat-log').scrollHeight;
          } else if (ev.type === 'retry') {
            fullText = '';
            buddyBubble.textContent = "one sec, gathering my thoughts...";
          } else if (ev.type === 'error') {
            buddyBubble.textContent = `Hmm, something went wrong: ${ev.message}`;
          } else if (ev.type === 'done') {
            fullText = ev.full_text || fullText;
            buddyBubble.textContent = fullText;
          }
        }
      }
      buddyBubble.classList.remove('streaming');
    } catch (_) {
      buddyBubble.textContent = "I couldn't reach you just now — try again?";
      buddyBubble.classList.remove('streaming');
    } finally {
      input.disabled = false;
      input.focus();
    }
  });
}

// ── Tasks ──────────────────────────────────────────────────────────────────

function renderTaskItem(task) {
  const li = document.createElement('li');
  li.className = `task-item${task.done ? ' done' : ''}`;
  li.dataset.id = task.id;

  const checkbox = document.createElement('input');
  checkbox.type = 'checkbox';
  checkbox.checked = task.done;
  checkbox.addEventListener('change', async () => {
    await api(`/api/tasks/${task.id}`, { method: 'PATCH', body: JSON.stringify({ done: checkbox.checked }) });
    li.classList.toggle('done', checkbox.checked);
  });

  const text = document.createElement('span');
  text.className = 'task-text';
  text.textContent = task.text;

  li.appendChild(checkbox);
  li.appendChild(text);

  if (task.source === 'ai') {
    const badge = document.createElement('span');
    badge.className = 'ai-badge';
    badge.textContent = 'buddy';
    li.appendChild(badge);
  }

  const del = document.createElement('button');
  del.className = 'del-btn';
  del.textContent = '✕';
  del.addEventListener('click', async () => {
    await api(`/api/tasks/${task.id}`, { method: 'DELETE' });
    li.remove();
  });
  li.appendChild(del);

  return li;
}

async function loadTasks() {
  const tasks = await api('/api/tasks');
  const list = $('#task-list');
  list.innerHTML = '';
  tasks.slice().reverse().forEach((t) => list.appendChild(renderTaskItem(t)));
}

function wireTasks() {
  $('#task-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const input = $('#task-input');
    const text = input.value.trim();
    if (!text) return;
    input.value = '';
    const task = await api('/api/tasks', { method: 'POST', body: JSON.stringify({ text }) });
    $('#task-list').prepend(renderTaskItem(task));
  });

  $('#breakdown-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const input = $('#breakdown-input');
    const goal = input.value.trim();
    if (!goal) return;
    const button = e.target.querySelector('button');
    button.disabled = true;
    button.textContent = 'thinking...';
    try {
      const res = await api('/api/tasks/breakdown', { method: 'POST', body: JSON.stringify({ goal }) });
      input.value = '';
      (res.tasks || []).forEach((t) => $('#task-list').prepend(renderTaskItem(t)));
    } catch (_) {
      alert("Couldn't break that down right now — try again in a bit.");
    } finally {
      button.disabled = false;
      button.textContent = 'Break it down';
    }
  });
}

// ── Check-in ──────────────────────────────────────────────────────────────

function renderMoodEntry(entry) {
  const li = document.createElement('li');
  li.className = 'mood-entry';
  li.innerHTML = `<span>${entry.mood}</span><span class="mood-date">${entry.date}</span><span>${entry.note || ''}</span>`;
  return li;
}

async function loadMood() {
  const res = await api('/api/mood');
  $('#checkin-streak').textContent = res.streak;
  const history = $('#mood-history');
  history.innerHTML = '';
  res.entries.slice().reverse().forEach((e) => history.appendChild(renderMoodEntry(e)));
}

function wireCheckin() {
  $all('.mood-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      $all('.mood-btn').forEach((b) => b.classList.remove('selected'));
      btn.classList.add('selected');
      state.moodSelected = btn.dataset.mood;
    });
  });

  $('#mood-submit').addEventListener('click', async () => {
    if (!state.moodSelected) {
      alert('Pick a mood first! 🌷');
      return;
    }
    const note = $('#mood-note').value.trim();
    const res = await api('/api/mood', { method: 'POST', body: JSON.stringify({ mood: state.moodSelected, note }) });
    $('#checkin-streak').textContent = res.streak;
    $('#mood-history').prepend(renderMoodEntry(res.entry));
    $('#mood-note').value = '';
    $all('.mood-btn').forEach((b) => b.classList.remove('selected'));
    state.moodSelected = null;
    if (res.streak > 0) {
      $('#streak-badge').classList.remove('hidden');
      $('#streak-count').textContent = res.streak;
    }
  });
}

// ── Memory ─────────────────────────────────────────────────────────────────

function renderProfileMarkdown(text) {
  const lines = text.split('\n');
  let html = '';
  let inList = false;
  for (const line of lines) {
    if (line.startsWith('# ')) continue; // title shown via topbar/greeting already
    if (line.startsWith('_')) continue;  // last-updated stamp
    const heading = line.match(/^##\s+(.+)$/);
    const bullet = line.match(/^-\s+(.+)$/);
    if (heading) {
      if (inList) { html += '</ul>'; inList = false; }
      html += `<h3>${heading[1]}</h3>`;
    } else if (bullet) {
      if (!inList) { html += '<ul>'; inList = true; }
      html += `<li>${bullet[1]}</li>`;
    } else if (line.trim().startsWith('_')) {
      // italic placeholder line, skip
    }
  }
  if (inList) html += '</ul>';
  return html;
}

async function loadMemory() {
  const res = await api('/api/profile');
  state.profileText = res.text;
  $('#memory-view').innerHTML = renderProfileMarkdown(res.text);
  $('#memory-edit').value = res.text;
}

function wireMemory() {
  $('#memory-edit-toggle').addEventListener('click', () => {
    $('#memory-view').classList.add('hidden');
    $('#memory-edit').classList.remove('hidden');
    $('#memory-edit-toggle').classList.add('hidden');
    $('#memory-save-btn').classList.remove('hidden');
  });

  $('#memory-save-btn').addEventListener('click', async () => {
    const text = $('#memory-edit').value;
    await api('/api/profile', { method: 'PUT', body: JSON.stringify({ text }) });
    state.profileText = text;
    $('#memory-view').innerHTML = renderProfileMarkdown(text);
    $('#memory-view').classList.remove('hidden');
    $('#memory-edit').classList.add('hidden');
    $('#memory-edit-toggle').classList.remove('hidden');
    $('#memory-save-btn').classList.add('hidden');
  });
}

boot();
