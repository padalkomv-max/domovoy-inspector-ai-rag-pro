const messages = document.querySelector('#messages');
const form = document.querySelector('#chat-form');
const input = document.querySelector('#message-input');
const sendButton = document.querySelector('#send-button');
const typing = document.querySelector('#typing');
const errorBox = document.querySelector('#error');
const charCount = document.querySelector('#char-count');
const history = [];
const REQUEST_TIMEOUT_MS = 75000;

async function readApiResponse(response) {
  const raw = await response.text();
  if (!raw.trim()) {
    return {detail: `Сервер вернул пустой ответ (HTTP ${response.status}). Повторите запрос позднее.`};
  }
  try {
    const data = JSON.parse(raw);
    return data && typeof data === 'object'
      ? data
      : {detail: `Сервер вернул некорректный JSON (HTTP ${response.status}).`};
  } catch {
    return {detail: `Сервер вернул некорректный JSON (HTTP ${response.status}).`};
  }
}

function escapeHtml(value) {
  return value.replace(/[&<>'"]/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#039;','"':'&quot;'}[char]));
}

function renderText(value) {
  return escapeHtml(value).replace(/\n/g, '<br>');
}

function addMessage(role, text, sources = [], diagnostics = null) {
  const item = document.createElement('article');
  item.className = `message ${role}`;
  const label = role === 'user' ? 'ВЫ' : 'ОПЕРАТОР · RAG PRO';
  let html = `<div class="message-label">${label}</div><div class="bubble">${renderText(text)}</div>`;
  if (role === 'assistant' && sources.length) {
    html += `<div class="sources"><span class="sources-title">Источники</span>${sources.map(source => `<span class="source-chip"><b>${escapeHtml(source.source)}</b><small>${escapeHtml(source.section)} · v${escapeHtml(source.version)}</small></span>`).join('')}</div>`;
  }
  if (role === 'assistant' && diagnostics) {
    html += `<details class="diagnostics"><summary>Диагностика поиска</summary><div class="diag-grid"><div><b>Векторный поиск</b><span>${diagnostics.vector_results.length} результатов</span></div><div><b>BM25</b><span>${diagnostics.bm25_results.length} результатов</span></div><div><b>RRF</b><span>top-${diagnostics.final_chunks.length}, k=${diagnostics.rrf_k}</span></div></div><div class="diag-list"><h4>Векторный поиск</h4>${diagnostics.vector_results.map(row => `<p><code>${row.score}</code> ${escapeHtml(row.source)} · ${escapeHtml(row.section)}</p>`).join('')}<h4>BM25</h4>${diagnostics.bm25_results.map(row => `<p><code>${row.score}</code> ${escapeHtml(row.source)} · ${escapeHtml(row.section)}</p>`).join('')}<h4>Передано LLM</h4>${diagnostics.final_chunks.map((row, i) => `<p><code>${i + 1}</code> ${escapeHtml(row.source)} · ${escapeHtml(row.section)}</p>`).join('')}</div></details>`;
  }
  if (role === 'assistant') {
    html += '<div class="message-actions"><button class="export-button" type="button" data-export-markdown>Скачать Markdown</button></div>';
  }
  item.innerHTML = html;
  const exportButton = item.querySelector('[data-export-markdown]');
  if (exportButton) {
    exportButton.addEventListener('click', () => {
      const stamp = new Date().toISOString().replace(/[:.]/g, '-');
      const markdown = buildMarkdownExport(text, sources, diagnostics);
      downloadMarkdown(`domovoy-inspector-answer-${stamp}.md`, markdown);
    });
  }
  messages.appendChild(item);
  item.scrollIntoView({behavior: 'smooth', block: 'nearest'});
}

function setStatus(ready, detail) {
  document.querySelector('#status-dot').classList.toggle('offline', !ready);
  document.querySelector('#status-title').textContent = ready ? 'База знаний готова' : 'Индекс не готов';
  document.querySelector('#status-detail').textContent = detail;
}

async function checkStatus() {
  try {
    const response = await fetch('/api/knowledge/status');
    const data = await readApiResponse(response);
    if (!response.ok) throw new Error(data.detail || `Сервер вернул HTTP ${response.status}.`);
    setStatus(data.ready, data.ready ? `${data.documents} документов · ${data.chunks} чанков` : 'Запустите переиндексацию');
  } catch (error) { setStatus(false, error.message || 'Сервер недоступен'); }
}

function updateInput() {
  const length = input.value.length;
  charCount.textContent = `${length} / 2000`;
  sendButton.disabled = !input.value.trim();
  input.style.height = 'auto';
  input.style.height = `${Math.min(input.scrollHeight, 150)}px`;
}

async function ask(question) {
  if (!question.trim()) return;
  addMessage('user', question);
  input.value = '';
  updateInput();
  errorBox.hidden = true;
  typing.hidden = false;
  sendButton.disabled = true;
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  try {
    const response = await fetch('/api/chat', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({message: question, history}), signal: controller.signal});
    const data = await readApiResponse(response);
    if (!response.ok) throw new Error(data.detail || 'Не удалось получить ответ.');
    if (typeof data.reply !== 'string' || !data.reply.trim()) throw new Error('Сервер вернул неполный ответ без текста оператора.');
    history.push({role: 'user', content: question}, {role: 'assistant', content: data.reply});
    while (history.length > 12) history.shift();
    addMessage('assistant', data.reply, data.sources, data.diagnostics);
  } catch (error) {
    errorBox.textContent = error.name === 'AbortError'
      ? 'Сервер не ответил вовремя. Возможно, Render загружает модель или перезапускает сервис. Повторите запрос через несколько секунд.'
      : error instanceof TypeError
        ? 'Не удалось связаться с сервером. Проверьте доступность Render и повторите запрос.'
      : error.message || 'Не удалось связаться с сервером.';
    errorBox.hidden = false;
  } finally { clearTimeout(timeoutId); typing.hidden = true; updateInput(); input.focus(); }
}

form.addEventListener('submit', event => { event.preventDefault(); ask(input.value); });
input.addEventListener('input', updateInput);
input.addEventListener('keydown', event => { if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); form.requestSubmit(); } });
document.querySelectorAll('[data-question]').forEach(button => button.addEventListener('click', () => ask(button.dataset.question)));
document.querySelector('#clear-chat').addEventListener('click', () => { messages.innerHTML = ''; history.length = 0; errorBox.hidden = true; });
checkStatus();
updateInput();
