const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const root = path.join(__dirname, '..');
const html = fs.readFileSync(path.join(root, 'frontend', 'index.html'), 'utf8');
const js = fs.readFileSync(path.join(root, 'frontend', 'app.js'), 'utf8');

test('frontend exposes Russian chat and RAG diagnostics', () => {
  assert.ok(js.includes('Диагностика поиска'));
  assert.match(html, /Домовой инспектор/);
  assert.ok(js.includes("/api/chat"));
  assert.match(js, /vector_results/);
  assert.match(js, /bm25_results/);
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
