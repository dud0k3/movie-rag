const state = { selected: null };
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

async function discover() {
  const query = $('discover-input').value.trim();
  if (query.length < 2) return;
  const button = $('discover-button');
  button.disabled = true;
  $('discover-results').innerHTML = '<p class="muted">Ищем в каталоге TMDB…</p>';
  try {
    const data = await getJSON(`/discover?q=${encodeURIComponent(query)}`);
    const results = data.results.slice(0, 6);
    $('discover-results').innerHTML = results.length ? results.map((item, index) => `
      <button class="discovery-item" data-index="${index}">
        <span><strong>${escapeHtml(item.title)}</strong><small>${item.type === 'movie' ? 'Фильм' : item.type === 'tv' ? 'Сериал' : 'Человек'}${item.year ? ' · ' + escapeHtml(item.year) : ''}</small></span>
        <span class="item-arrow" aria-hidden="true">→</span>
      </button>`).join('') : '<p class="muted">Ничего не найдено. Попробуйте оригинальное название.</p>';
    document.querySelectorAll('.discovery-item').forEach(button => button.addEventListener('click', () => select(results[Number(button.dataset.index)])));
  } catch (error) {
    $('discover-results').innerHTML = `<p class="error">${escapeHtml(error.message)}</p>`;
  } finally { button.disabled = false; }
}

async function select(item) {
  state.selected = item;
  $('discover-results').innerHTML = '';
  $('selected').classList.remove('hidden');
  const typeLabel = item.type === 'movie' ? 'Фильм' : item.type === 'tv' ? 'Сериал' : 'Человек';
  $('selected').innerHTML = `<span>${typeLabel}: ${escapeHtml(item.title)}</span><button id="clear-selected" aria-label="Сбросить выбор">×</button>`;
  $('clear-selected').addEventListener('click', () => { state.selected = null; $('selected').classList.add('hidden'); });
  $('answer-panel').classList.add('hidden');
  $('sources-column').classList.add('hidden');
  document.querySelector('.workspace').classList.remove('has-answer');
  $('question').placeholder = item.type === 'movie' ? `Что известно о создании фильма «${item.title}»?` : item.type === 'tv' ? `Спросите о сериале «${item.title}» или его героях` : `Спросите о человеке ${item.title}`;
  setStatus('Загружаем материалы. Первый запрос может занять немного времени…');
  try {
    await getJSON('/ingest', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({type:item.type,id:item.id}) });
    setStatus('Материалы готовы. Задайте вопрос.');
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
  if (query.length < 2) { setStatus('Введите вопрос.', true); return; }
  document.querySelector('.advanced').open = false;
  const button = $('ask-button');
  button.disabled = true;
  setStatus('Ищем фрагменты и готовим ответ…');
  try {
    const params = new URLSearchParams({q:query,mode:$('search-mode').value});
    if (state.selected) { params.set('entity_type',state.selected.type); params.set('entity_id',state.selected.id); }
    const data = await getJSON(`/ask?${params}`);
    $('answer-panel').classList.remove('hidden');
    $('sources-column').classList.remove('hidden');
    document.querySelector('.workspace').classList.add('has-answer');
    $('answer-title').textContent = 'Ответ';
    $('answer-mode').textContent = data.answer_mode === 'qwen' ? 'Qwen · локальная модель' : data.answer_mode === 'structured' ? 'Ответ по данным каталога' : 'Выжимка из источников';
    const sources = uniqueSources(data.sources);
    renderAnswer(data.answer, sources.citations);
    renderSources(sources.list);
    setStatus(data.semantic_error || 'Ответ готов.', Boolean(data.semantic_error));
    $('answer-panel').scrollIntoView({behavior:'smooth',block:'nearest'});
  } catch(error) { setStatus(error.message, true); }
  finally { button.disabled = false; }
}

$('discover-button').addEventListener('click', discover);
$('discover-input').addEventListener('keydown', event => { if (event.key === 'Enter') discover(); });
$('ask-button').addEventListener('click', ask);
$('question').addEventListener('keydown', event => { if (event.key === 'Enter' && (event.metaKey || event.ctrlKey)) ask(); });
