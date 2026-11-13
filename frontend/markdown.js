function markdownCell(value) {
  return String(value ?? '—').replace(/\|/g, '\\|').replace(/\r?\n/g, ' ');
}

function diagnosticRows(rows) {
  return Array.isArray(rows) ? rows : [];
}

function isUnsupportedAnswer(reply) {
  const normalized = String(reply ?? '').toLocaleLowerCase('ru-RU');
  return normalized.includes('в базе знаний нет подтверждённой информации')
    || normalized.includes('источники: нет подтверждённого фрагмента');
}

function sourceVersion(row, confirmedSources) {
  const match = confirmedSources.find(source => source.source === row.source && source.section === row.section)
    || confirmedSources.find(source => source.source === row.source);
  return row.version ?? match?.version ?? '—';
}

function buildMarkdownExport(reply, sources = [], diagnostics = null) {
  const candidateSources = Array.isArray(sources) ? sources : [];
  const refusedForMissingInformation = isUnsupportedAnswer(reply);
  const confirmedSources = refusedForMissingInformation ? [] : candidateSources;
  const lines = [
    '# Ответ AI-оператор',
    '',
    '## Ответ',
    '',
    String(reply ?? '').trim() || 'Ответ отсутствует.',
    '',
    '## Источники',
    '',
  ];

  if (confirmedSources.length) {
    lines.push('| Источник | Раздел | Версия |', '|---|---|---|');
    confirmedSources.forEach(source => {
      lines.push(`| ${markdownCell(source.source)} | ${markdownCell(source.section)} | ${markdownCell(source.version)} |`);
    });
  } else {
    lines.push('**Подтверждённый источник:** отсутствует в базе знаний.');
    if (refusedForMissingInformation && candidateSources.length) {
      lines.push('', 'Поиск вернул кандидатов, но ответ оператора не подтверждает их как источники по вопросу.');
    }
  }

  lines.push('', '## Диагностика поиска', '');
  if (!diagnostics) {
    lines.push('Диагностика поиска отсутствует.');
    return `${lines.join('\n')}\n`;
  }

  const vectorResults = diagnosticRows(diagnostics.vector_results);
  const bm25Results = diagnosticRows(diagnostics.bm25_results);
  const finalChunks = diagnosticRows(diagnostics.final_chunks);
  lines.push(`RRF: top-${finalChunks.length}, k=${markdownCell(diagnostics.rrf_k)}`, '');
  lines.push('### FAISS', '', '| Ранг | Оценка | Источник | Раздел | Версия |', '|---:|---:|---|---|---|');
  vectorResults.forEach(row => {
    lines.push(`| ${markdownCell(row.rank)} | ${markdownCell(row.score)} | ${markdownCell(row.source)} | ${markdownCell(row.section)} | ${markdownCell(sourceVersion(row, confirmedSources))} |`);
  });
  if (!vectorResults.length) lines.push('| — | результатов нет | — | — | — |');

  lines.push('', '### BM25', '', '| Ранг | Оценка | Источник | Раздел | Версия |', '|---:|---:|---|---|---|');
  bm25Results.forEach(row => {
    lines.push(`| ${markdownCell(row.rank)} | ${markdownCell(row.score)} | ${markdownCell(row.source)} | ${markdownCell(row.section)} | ${markdownCell(sourceVersion(row, confirmedSources))} |`);
  });
  if (!bm25Results.length) lines.push('| — | результатов нет | — | — | — |');

  lines.push('', '### RRF: фрагменты, переданные LLM', '', '| Позиция | Источник | Раздел | Версия | RRF score |', '|---:|---|---|---|---:|');
  finalChunks.forEach((row, index) => {
    lines.push(`| ${index + 1} | ${markdownCell(row.source)} | ${markdownCell(row.section)} | ${markdownCell(sourceVersion(row, confirmedSources))} | ${markdownCell(row.rrf_score)} |`);
  });
  if (!finalChunks.length) lines.push('| — | итоговых фрагментов нет | — | — | — |');
  return `${lines.join('\n')}\n`;
}

function downloadMarkdown(filename, content) {
  const blob = new Blob([content], {type: 'text/markdown;charset=utf-8'});
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = {buildMarkdownExport};
}
