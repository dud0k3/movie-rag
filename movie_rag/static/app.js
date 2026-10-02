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

async function updateStats() {
  try {
    const data = await getJSON('/health');
    $('catalog-status').textContent = `${data.database.movies} фильмов · ${data.database.people} персон`;
  } catch {
    $('catalog-status').textContent = 'Каталог недоступен';
  }
}

async function discover() {
  const query = $('discover-input').value.trim();
  if (query.length < 2) return;
  const button = $('discover-button');
  button.disabled = true;
  $('discover-results').innerHTML = '<p class="muted">Ищем в каталоге TMDB…</p>';
  try {
    const data = await getJSON(`/discover?q=${encodeURIComponent(query)}`);
    const results = data.results.slice(0, 12);
    $('discover-results').innerHTML = results.length ? results.map((item, index) => `
      <button class="discovery-item" data-index="${index}">
        <span class="item-icon">${item.type === 'movie' ? '▣' : '◉'}</span>
        <span><strong>${escapeHtml(item.title)}</strong><small>${item.type === 'movie' ? 'Фильм' : 'Человек'}${item.year ? ' · ' + escapeHtml(item.year) : ''}</small></span>
        <span class="item-arrow">↗</span>
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
  $('selected').innerHTML = `<div><small>ВЫБРАНО · ${item.type === 'movie' ? 'ФИЛЬМ' : 'ЧЕЛОВЕК'}</small><strong>${escapeHtml(item.title)}</strong></div><button id="clear-selected" aria-label="Сбросить выбор">×</button>`;
  $('clear-selected').addEventListener('click', () => { state.selected = null; $('selected').classList.add('hidden'); });
  $('question').placeholder = item.type === 'movie' ? `Что известно о создании фильма «${item.title}»?` : `Какие фильмы связаны с ${item.title}?`;
  setStatus('Загружаем материалы. Первый запрос может занять немного времени…');
  try {
    await getJSON('/ingest', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({type:item.type,id:item.id}) });
    setStatus('Материалы готовы. Задайте вопрос.');
    updateStats();
  } catch(error) { setStatus(error.message, true); }
}

function renderSources(sources) {
  $('sources').innerHTML = sources.length ? sources.map(source => `
    <article class="source-card"><div class="source-meta"><span class="source-number">${source.number}</span><span>${escapeHtml(source.source)}</span></div>
      <h3><a href="${escapeHtml(source.url)}" target="_blank" rel="noopener noreferrer">${escapeHtml(source.title)} ↗</a></h3>
      <p>${source.source === 'Wikipedia' ? 'Англоязычный источник. Его сведения в ответе переведены на русский.' : escapeHtml(source.excerpt).slice(0, 240) + (source.excerpt.length > 240 ? '…' : '')}</p>
    </article>`).join('') : '<div class="empty-sources"><span>↗</span><p>Для этого вопроса источники пока не найдены.</p></div>';
}

async function ask() {
  const query = $('question').value.trim();
  if (query.length < 2) { setStatus('Введите вопрос.', true); return; }
  const button = $('ask-button');
  button.disabled = true;
  setStatus('Ищем фрагменты и готовим ответ…');
  try {
    const params = new URLSearchParams({q:query,mode:$('search-mode').value});
    if (state.selected) { params.set('entity_type',state.selected.type); params.set('entity_id',state.selected.id); }
    const data = await getJSON(`/ask?${params}`);
    $('answer-panel').classList.remove('hidden');
    $('answer-title').textContent = state.selected?.title || 'Что удалось найти';
    $('answer-mode').textContent = data.answer_mode === 'qwen' ? 'Qwen · локальная модель' : 'Выжимка из источников';
    $('answer-text').textContent = data.answer;
    renderSources(data.sources);
    setStatus(data.semantic_error || 'Ответ готов.', Boolean(data.semantic_error));
    $('answer-panel').scrollIntoView({behavior:'smooth',block:'nearest'});
  } catch(error) { setStatus(error.message, true); }
  finally { button.disabled = false; }
}

$('discover-button').addEventListener('click', discover);
$('discover-input').addEventListener('keydown', event => { if (event.key === 'Enter') discover(); });
$('ask-button').addEventListener('click', ask);
$('question').addEventListener('keydown', event => { if (event.key === 'Enter' && (event.metaKey || event.ctrlKey)) ask(); });
document.querySelectorAll('[data-question]').forEach(button => button.addEventListener('click', () => { $('question').value = button.dataset.question; $('question').focus(); }));
updateStats();
