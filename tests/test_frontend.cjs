const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const root = path.join(__dirname, '..');
const html = fs.readFileSync(path.join(root, 'frontend', 'index.html'), 'utf8');
const js = fs.readFileSync(path.join(root, 'frontend', 'app.js'), 'utf8');
const {buildMarkdownExport} = require(path.join(root, 'frontend', 'markdown.js'));

test('frontend exposes Russian chat and RAG diagnostics', () => {
  assert.ok(js.includes('Диагностика поиска'));
  assert.match(html, /Домовой инспектор/);
  assert.ok(js.includes("/api/chat"));
  assert.match(js, /vector_results/);
  assert.match(js, /bm25_results/);
  assert.match(js, /data-export-markdown/);
  assert.match(html, /markdown\.js/);
});

test('frontend does not contain an API key', () => {
  assert.doesNotMatch(js, /Bearer\s+sk-|LLM_API_KEY\s*=/i);
  assert.doesNotMatch(html, /LLM_API_KEY|sk-[A-Za-z0-9]{20,}/i);
});

test('frontend handles empty, invalid and timed-out API responses', () => {
  assert.match(js, /const raw = await response\.text\(\)/);
  assert.match(js, /JSON\.parse\(raw\)/);
  assert.match(js, /Сервер вернул пустой ответ/);
  assert.match(js, /Сервер вернул некорректный JSON/);
  assert.match(js, /AbortController/);
  assert.match(js, /Сервер не ответил вовремя/);
  assert.match(js, /Не удалось связаться с сервером/);
  assert.doesNotMatch(js, /const data = await response\.json\(\)/);
});

test('markdown export contains answer, source versions and diagnostics', () => {
  const markdown = buildMarkdownExport(
    'Ответ по документации',
    [{source: '02_создание_обращения.md', section: 'Начало', version: '1.0'}],
    {
      vector_results: [{rank: 1, score: 0.91, source: '02_создание_обращения.md', section: 'Начало'}],
      bm25_results: [{rank: 1, score: 3.2, source: '02_создание_обращения.md', section: 'Начало'}],
      final_chunks: [{source: '02_создание_обращения.md', section: 'Начало', rrf_score: 0.032}],
      rrf_k: 60,
    },
  );
  assert.match(markdown, /Ответ по документации/);
  assert.match(markdown, /02_создание_обращения\.md/);
  assert.match(markdown, /Версия/);
  assert.match(markdown, /1\.0/);
  assert.match(markdown, /FAISS/);
  assert.match(markdown, /BM25/);
  assert.match(markdown, /RRF/);
});

test('markdown export marks refusal without confirmed source', () => {
  const markdown = buildMarkdownExport(
    'В документации нет подтверждённой информации.',
    [],
    {vector_results: [], bm25_results: [], final_chunks: [], rrf_k: 60},
  );
  assert.match(markdown, /Подтверждённый источник.*отсутствует/);
  assert.match(markdown, /В документации нет подтверждённой информации/);
  assert.doesNotMatch(markdown, /undefined/);
});

test('markdown export does not confirm irrelevant candidates after refusal', () => {
  const markdown = buildMarkdownExport(
    'В базе знаний нет подтверждённой информации по этому вопросу.\nИсточники: нет подтверждённого фрагмента',
    [{source: '09_ошибки_и_диагностика.md', section: 'Случайный раздел', version: '1.0'}],
    {
      vector_results: [{rank: 1, score: 0.8, source: '09_ошибки_и_диагностика.md', section: 'Случайный раздел'}],
      bm25_results: [{rank: 1, score: 0, source: '09_ошибки_и_диагностика.md', section: 'Случайный раздел'}],
      final_chunks: [{source: '09_ошибки_и_диагностика.md', section: 'Случайный раздел', version: '1.0', rrf_score: 0.02}],
      rrf_k: 60,
    },
  );
  const sourcesSection = markdown.split('## Диагностика поиска', 1)[0];
  assert.match(sourcesSection, /Подтверждённый источник.*отсутствует/);
  assert.doesNotMatch(sourcesSection, /09_ошибки_и_диагностика\.md/);
  assert.match(markdown, /Поиск вернул кандидатов/);
  assert.match(markdown, /### RRF/);
});
