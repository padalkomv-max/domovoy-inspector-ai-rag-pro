const messages = document.querySelector('#messages');
const form = document.querySelector('#chat-form');
const input = document.querySelector('#message-input');
const sendButton = document.querySelector('#send-button');
const typing = document.querySelector('#typing');
const errorBox = document.querySelector('#error');
const charCount = document.querySelector('#char-count');
const history = [];

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
  item.innerHTML = html;
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
    const data = await response.json();
    setStatus(data.ready, data.ready ? `${data.documents} документов · ${data.chunks} чанков` : 'Запустите переиндексацию');
  } catch { setStatus(false, 'Сервер недоступен'); }
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
  try {
    const response = await fetch('/api/chat', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({message: question, history})});
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'Не удалось получить ответ.');
    history.push({role: 'user', content: question}, {role: 'assistant', content: data.reply});
    while (history.length > 12) history.shift();
    addMessage('assistant', data.reply, data.sources, data.diagnostics);
  } catch (error) {
    errorBox.textContent = error.message;
    errorBox.hidden = false;
  } finally { typing.hidden = true; updateInput(); input.focus(); }
}

form.addEventListener('submit', event => { event.preventDefault(); ask(input.value); });
input.addEventListener('input', updateInput);
input.addEventListener('keydown', event => { if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); form.requestSubmit(); } });
document.querySelectorAll('[data-question]').forEach(button => button.addEventListener('click', () => ask(button.dataset.question)));
document.querySelector('#clear-chat').addEventListener('click', () => { messages.innerHTML = ''; history.length = 0; errorBox.hidden = true; });
checkStatus();
updateInput();
