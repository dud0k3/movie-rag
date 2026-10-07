const state = { selected: null, results: [], visibleItems: [], filter: 'all', activeIndex: -1,
  loading: false, searchTimer: null, searchController: null, searchSeq: 0,
  recent: (() => { try { return JSON.parse(localStorage.getItem('movie-rag-recent') || '[]'); } catch { return []; } })() };
const $ = id => document.getElementById(id);
const escapeHtml = value => String(value ?? '').replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));

function setStatus(message, error = false) {
  $('status').textContent = message;
  $('status').classList.toggle('error', error);
}

async function getJSON(url, options) {
  const response = await fetch(url, options);
  let data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.detail || `Ошибка ${response.status}`);
  return data;
}

const kindLabel = type => type === 'movie' ? 'Фильм' : type === 'tv' ? 'Сериал' : 'Человек';

function closePicker() {
  $('discover-results').classList.add('hidden');
  $('discover-input').setAttribute('aria-expanded', 'false');
  $('discover-input').removeAttribute('aria-activedescendant');
  state.activeIndex = -1;
}

function renderPicker() {
  const query = $('discover-input').value.trim();
  const recentMode = query.length < 2;
  const items = recentMode ? state.recent : state.results;
  const filtered = state.filter === 'all' || recentMode ? items : items.filter(item => item.type === state.filter);
  state.visibleItems = filtered.slice(0, 8);
  const tabs = !recentMode && items.length ? `<div class="filter-row" aria-label="Тип результата">${[
    ['all', 'Все'], ['movie', 'Фильмы'], ['tv', 'Сериалы'], ['person', 'Люди']
  ].map(([value, label]) => `<button type="button" class="filter-chip ${state.filter === value ? 'active' : ''}" data-filter="${value}" aria-pressed="${state.filter === value}">${label}</button>`).join('')}</div>` : '';
  let body = '';
  if (state.loading && !recentMode) body = '<p class="picker-note">Ищем…</p>';
  else if (!state.visibleItems.length) body = `<p class="picker-note">${recentMode ? 'Начните вводить название или имя' : 'Ничего не найдено'}</p>`;
  else body = `<div id="option-list" role="listbox" aria-label="Варианты">${state.visibleItems.map((item, index) => {
    const fallback = `<span class="poster-fallback" aria-hidden="true">${escapeHtml((item.title || '✦').slice(0, 1).toUpperCase())}</span>`;
    const image = fallback + (item.poster_path && /^\/[a-zA-Z0-9._-]+$/.test(item.poster_path) ? `<img src="https://image.tmdb.org/t/p/w92/${encodeURIComponent(item.poster_path.slice(1))}" alt="" loading="lazy">` : '');
    const name = escapeHtml(item.title);
    const subtitle = [kindLabel(item.type), item.year].filter(Boolean).join(' · ');
    return `<button type="button" class="discovery-item ${state.activeIndex === index ? 'active' : ''}" id="option-${index}" role="option" aria-selected="${state.activeIndex === index}" data-index="${index}"><span class="poster">${image}</span><span class="result-copy"><strong>${name}</strong><small>${escapeHtml(subtitle)}</small></span><span class="item-arrow" aria-hidden="true">↗</span></button>`;
  }).join('')}</div>`;
  $('discover-results').innerHTML = (recentMode && state.recent.length ? '<div class="picker-caption">Недавние</div>' : '') + tabs + body;
  $('discover-results').classList.remove('hidden');
  $('discover-input').setAttribute('aria-expanded', 'true');
  document.querySelectorAll('.discovery-item').forEach(button => button.addEventListener('click', () => select(state.visibleItems[Number(button.dataset.index)])));
  document.querySelectorAll('.poster img').forEach(img => {
    if (img.complete && img.naturalWidth) img.classList.add('loaded');
    img.addEventListener('load', () => img.classList.add('loaded'));
    img.addEventListener('error', () => img.remove());
  });
  document.querySelectorAll('.filter-chip').forEach(button => button.addEventListener('click', () => {
    state.filter = button.dataset.filter;
    state.activeIndex = -1;
    renderPicker();
  }));
}

async function discover() {
  const query = $('discover-input').value.trim();
  if (query.length < 2) { renderPicker(); return; }
  state.searchController?.abort();
  state.searchController = new AbortController();
  const sequence = ++state.searchSeq;
  state.loading = true;
  renderPicker();
  try {
    const data = await getJSON(`/discover?q=${encodeURIComponent(query)}`, { signal: state.searchController.signal });
    if (sequence !== state.searchSeq) return;
    state.results = data.results;
    state.activeIndex = -1;
  } catch (error) {
    if (error.name === 'AbortError' || sequence !== state.searchSeq) return;
    state.results = [];
    $('discover-results').innerHTML = `<p class="picker-note error">${escapeHtml(error.message)}</p>`;
    return;
  } finally {
    if (sequence === state.searchSeq) { state.loading = false; renderPicker(); }
  }
}

async function select(item) {
  if (!item) return;
  clearTimeout(state.searchTimer);
  state.searchController?.abort();
  state.searchSeq++;
  state.loading = false;
  state.selected = item;
  state.recent = [item, ...state.recent.filter(old => old.type !== item.type || old.id !== item.id)].slice(0, 4);
  try { localStorage.setItem('movie-rag-recent', JSON.stringify(state.recent)); } catch {}
  $('discover-input').value = item.title;
  closePicker();
  $('selected').classList.remove('hidden');
  $('selected').innerHTML = `<span>${escapeHtml(item.title)}</span><button id="clear-selected" aria-label="Сбросить выбор">×</button>`;
  $('clear-selected').addEventListener('click', () => {
    state.selected = null;
    $('selected').classList.add('hidden');
    $('discover-input').value = '';
    $('question').placeholder = 'Спросите о фильме или человеке…';
    $('answer-panel').classList.add('hidden');
    $('sources-column').classList.add('hidden');
    $('discover-input').focus();
    renderPicker();
  });
  $('answer-panel').classList.add('hidden');
  $('sources-column').classList.add('hidden');
  document.querySelector('.workspace').classList.remove('has-answer');
  $('question').placeholder = item.type === 'movie' ? `Спросите о фильме «${item.title}»` : item.type === 'tv' ? `Спросите о сериале «${item.title}»` : `Спросите о ${item.title}`;
  setStatus('Загружаем материалы…');
  try {
    await getJSON('/ingest', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({type:item.type,id:item.id}) });
    setStatus('Можно задавать вопрос.');
  } catch(error) { setStatus(error.message, true); }
}

function renderSources(sources) {
  const count = sources.length;
  $('source-count').textContent = `${count} ${count === 1 ? 'источник' : count < 5 ? 'источника' : 'источников'}`;
  $('sources').innerHTML = sources.length ? sources.map(source => `
    <article class="source-card" id="source-${source.number}"><a href="${escapeHtml(source.url)}" target="_blank" rel="noopener noreferrer">
      <span class="source-meta"><span class="source-number">${source.number}</span><span>${escapeHtml(source.source)}</span></span>
      <span class="source-title">${escapeHtml(source.title)}</span>
      <span class="source-url">${escapeHtml(new URL(source.url).hostname.replace(/^www\./, ''))}</span>
    </a></article>`).join('') : '<p class="muted">Источники не найдены.</p>';
}

function uniqueSources(sources) {
  const byUrl = new Map();
  const list = [];
  const citations = {};
  for (const source of sources) {
    if (!byUrl.has(source.url)) {
      const number = list.length + 1;
      byUrl.set(source.url, number);
      list.push({ ...source, number });
    }
    citations[source.number] = byUrl.get(source.url);
  }
  return { list, citations };
}

function renderAnswer(answer, citations) {
  const normalized = answer.trim()
    .replace(/\[(\d+)\]/g, (marker, number) => citations[number] ? `[${citations[number]}]` : marker)
    .replace(/(\[\d+\])(?:\s*\1)+/g, '$1');
  const sourceNumbers = new Set(Object.values(citations));
  $('answer-text').innerHTML = normalized.split(/\n\s*\n/).map(paragraph => `<p>${escapeHtml(paragraph).replace(/\[(\d+)\]/g, (marker, number) => sourceNumbers.has(Number(number)) ? `<a class="citation" href="#source-${number}" aria-label="Источник ${number}">[${number}]</a>` : marker).replace(/\n/g, '<br>')}</p>`).join('');
}

async function ask() {
  const query = $('question').value.trim();
  if ($('ask-button').disabled) { setStatus('Подготавливаем быстрый поиск…'); return; }
  if (query.length < 2) { setStatus('Введите вопрос.', true); return; }
  document.querySelector('.advanced').open = false;
  const button = $('ask-button');
  button.disabled = true;
  setStatus('Ищем фрагменты и готовим ответ…');
  try {
    const params = new URLSearchParams({q:query,mode:$('search-mode').value});
    if (state.selected) { params.set('entity_type',state.selected.type); params.set('entity_id',state.selected.id); }
    const response = await fetch(`/ask/stream?${params}`);
    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.detail || `Ошибка ${response.status}`);
    }
    if (!response.body) throw new Error('Браузер не поддерживает потоковые ответы.');
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let pending = '';
    let draft = '';
    let citations = {};
    let sourceList = [];
    let finished = false;
    const showCitedSources = answer => {
      const used = new Set([...answer.matchAll(/\[(\d+)\]/g)].map(match => citations[match[1]]).filter(Boolean));
      const visible = sourceList.filter(source => used.has(source.number));
      renderSources(visible);
      $('sources-column').classList.toggle('hidden', visible.length === 0);
    };
    $('answer-panel').classList.remove('hidden');
    document.querySelector('.workspace').classList.add('has-answer');
    $('answer-title').textContent = 'Ответ';
    $('answer-text').textContent = '';
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      pending += decoder.decode(value, {stream:true});
      let cut;
      while ((cut = pending.indexOf('\n')) !== -1) {
        const line = pending.slice(0, cut);
        pending = pending.slice(cut + 1);
        if (!line.trim()) continue;
        const event = JSON.parse(line);
        if (event.type === 'sources') {
          const sources = uniqueSources(event.sources);
          citations = sources.citations;
          sourceList = sources.list;
          setStatus('Формируем ответ…');
        } else if (event.type === 'delta') {
          draft += event.text;
          renderAnswer(draft, citations);
          showCitedSources(draft);
        } else if (event.type === 'done') {
          finished = true;
          renderAnswer(event.answer, citations);
          showCitedSources(event.answer);
          $('answer-mode').textContent = event.answer_mode === 'qwen' ? 'Qwen · локальная модель' : event.answer_mode === 'structured' ? 'Ответ по данным каталога' : 'Выжимка из источников';
          setStatus('Ответ готов.');
        }
      }
    }
    if (!finished) throw new Error('Соединение прервалось до завершения ответа.');
    $('answer-panel').scrollIntoView({behavior:'smooth',block:'nearest'});
  } catch(error) { setStatus(error.message, true); }
  finally { button.disabled = false; }
}

$('discover-input').addEventListener('focus', renderPicker);
$('discover-input').addEventListener('click', () => {
  if ($('discover-results').classList.contains('hidden')) renderPicker();
});
$('discover-input').addEventListener('input', () => {
  clearTimeout(state.searchTimer);
  state.searchController?.abort();
  state.searchSeq++;
  state.selected = null;
  $('selected').classList.add('hidden');
  $('question').placeholder = 'Спросите о фильме или человеке…';
  $('answer-panel').classList.add('hidden');
  $('sources-column').classList.add('hidden');
  state.filter = 'all';
  state.results = [];
  state.loading = $('discover-input').value.trim().length >= 2;
  renderPicker();
  state.searchTimer = window.setTimeout(discover, 260);
});
$('discover-input').addEventListener('keydown', event => {
  if (event.key === 'Escape') { event.preventDefault(); closePicker(); return; }
  if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
    event.preventDefault();
    if ($('discover-results').classList.contains('hidden')) renderPicker();
    if (!state.visibleItems.length) return;
    state.activeIndex = (state.activeIndex + (event.key === 'ArrowDown' ? 1 : -1) + state.visibleItems.length) % state.visibleItems.length;
    document.querySelectorAll('.discovery-item').forEach((item, index) => {
      item.classList.toggle('active', index === state.activeIndex);
      item.setAttribute('aria-selected', String(index === state.activeIndex));
    });
    const active = $('option-' + state.activeIndex);
    $('discover-input').setAttribute('aria-activedescendant', active.id);
    active.scrollIntoView({block:'nearest'});
  } else if (event.key === 'Enter') {
    event.preventDefault();
    if (state.visibleItems.length && !$('discover-results').classList.contains('hidden')) select(state.visibleItems[Math.max(0, state.activeIndex)]);
    else discover();
  }
});
document.addEventListener('pointerdown', event => {
  if (!$('entity-picker').contains(event.target)) closePicker();
});
$('ask-button').addEventListener('click', ask);
$('question').addEventListener('keydown', event => { if (event.key === 'Enter' && (event.metaKey || event.ctrlKey)) ask(); });

async function waitForReady() {
  try {
    const data = await getJSON('/health');
    if (data.ready) {
      $('ask-button').disabled = false;
      setStatus('Можно задавать вопросы.');
      return;
    }
  } catch {}
  setStatus('Подготавливаем быстрый поиск…');
  window.setTimeout(waitForReady, 400);
}

waitForReady();
