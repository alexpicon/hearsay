// Author: Alex Picon <alexnpc@me.com>
const $ = id => document.getElementById(id);
const API = '/api/hearsay-compare';
let names = {}, unlocked = false, loading = false, modelsReady = false, submitting = false;
const MODES = {
  speed: {models:['ours'], button:'Run Speed', description:'Runs only our original detector for the quickest result.'},
  accuracy: {models:['ours_v2'], button:'Run Accuracy', description:'Runs only our v2 fusion for stronger mixed-domain detection.'},
  both: {models:['ours','ours_v2'], button:'Compare Speed and Accuracy', description:'Runs both of our detectors sequentially on the same recording.'}
};
function mode() { return document.querySelector('input[name=mode]:checked').value; }
function selectedModels() {
  return MODES[mode()].models;
}
function updateMode() {
  const choice = MODES[mode()];
  $('mode-description').textContent = choice.description;
  $('run').textContent = submitting ? 'Uploading…' : modelsReady ? choice.button : 'Loading detectors…';
  $('run').disabled = submitting || !modelsReady || !selectedModels().length;
}
for (const radio of document.querySelectorAll('input[name=mode]')) radio.addEventListener('change', updateMode);
updateMode();
function node(tag, text, cls) { const el = document.createElement(tag); el.textContent = text; if (cls) el.className = cls; return el; }
async function request(path, options = {}) {
  const response = await fetch(API + path, {...options, headers: {'X-Comparison-Key': $('key').value, ...options.headers}});
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || 'Request failed');
  return data;
}
async function refresh() {
  if (loading) return;
  loading = true;
  try {
    const jobs = await request('/jobs'); unlocked = true;
    $('results').replaceChildren();
    if (!jobs.length) $('results').append(node('p', 'No comparisons yet. Upload a recording to begin.'));
    for (const job of jobs) {
      const card = node('div', '', 'job');
      card.append(node('h3', job.filename || 'Earlier upload'), node('p', new Date(job.created * 1000).toLocaleString() + (job.expected_label && job.expected_label !== 'unknown' ? ' · Supplied label: ' + job.expected_label : ''), 'muted'), node('p', `${job.status.replaceAll('_', ' ')}${job.current ? ' · ' + names[job.current] : ''}`, 'status'));
      if (job.error) card.append(node('p', job.error));
      const wrap = node('div', '', 'table-wrap'), table = node('table');
      const header = node('tr', ''); for (const title of ['Detector', 'Interpretation', 'Raw score ↑ synthetic', 'Time', 'Run status']) header.append(node('th', title));
      table.append(header);
      for (const model of job.models) {
        const result = job.results.find(r => r.model === model), row = node('tr', '');
        let interpretation = '—';
        if (result?.status === 'complete' && result.score !== undefined) {
          const predicted = result.score >= 0.5 ? 'synthetic' : 'real';
          interpretation = 'Leans ' + predicted;
          if (job.expected_label && job.expected_label !== 'unknown') interpretation += predicted === job.expected_label ? ' · Matches label' : ' · Disagrees with label';
          if (result.analyzed_seconds < result.duration_seconds) interpretation += ' · ' + (result.coverage_policy || 'Partial audio coverage');
        }
        const values = [names[model] || model, interpretation, result?.score !== undefined ? result.score.toFixed(6) : '—', result ? `${result.seconds}s` : '—', result?.error || result?.status || (job.current === model ? 'running' : job.status === 'failed' ? 'not run' : 'waiting')];
        for (const value of values) row.append(node('td', value));
        table.append(row);
      }
      wrap.append(table); card.append(wrap); $('results').append(card);
    }
  } catch (error) { unlocked = false; $('message').textContent = error.message; }
  finally { loading = false; }
}
$('form').addEventListener('submit', async event => {
  event.preventDefault();
  const file = $('audio').files[0], models = selectedModels();
  if (!models.length || !file) { $('message').textContent = 'Choose a file and at least one detector.'; return; }
  if (file.size > 10 * 1024 * 1024) { $('message').textContent = 'Maximum upload is 10 MiB.'; return; }
  submitting = true; $('mode-picker').disabled = true; updateMode(); $('message').textContent = 'Uploading…';
  try {
    await request('/jobs', {method:'POST', headers:{'Content-Type':'application/octet-stream', 'X-Models':models.join(','), 'X-Audio-Extension':'.' + file.name.split('.').pop().toLowerCase(), 'X-Audio-Name':encodeURIComponent(file.name), 'X-Expected-Label':$('expected').value}, body:file});
    $('message').textContent = 'Queued. You can leave this page; the worker continues through server reloads.';
    await refresh();
  } catch (error) { $('message').textContent = error.message; }
  finally { submitting = false; $('mode-picker').disabled = false; updateMode(); }
});
$('refresh').addEventListener('click', refresh);
fetch(API + '/models').then(r => r.json()).then(data => {
  document.querySelector('.eyebrow').textContent = 'ONE RECORDING. ' + data.models.length + ' DETECTORS.';
  for (const model of data.models) {
    names[model.id] = model.name;
    const label = node('label', ''), input = document.createElement('input');
    input.type = 'checkbox'; input.name = 'model'; input.value = model.id; input.checked = true; input.addEventListener('change', updateMode);
    label.append(input, document.createTextNode(' ' + model.name)); $('models').append(label);
  }
  for (const radio of document.querySelectorAll('input[name=mode]')) {
    radio.disabled = (MODES[radio.value].models || []).some(id => !names[id]);
  }
  if (document.querySelector('input[name=mode]:checked').disabled) {
    document.querySelector('input[name=mode]:not(:disabled)').checked = true;
  }
  modelsReady = true; updateMode();
}).catch(() => { $('message').textContent = 'Could not load detector list. Retry after the server is available.'; });
setInterval(() => { if (unlocked && $('key').value) refresh(); }, 2000);

// The catalog supplies labels and provenance only; every score is computed fresh.
let sampleCatalog = [], visibleSamples = [];
function currentSample() { return visibleSamples.find(s => s.id === $('sample-select').value); }
function showSample() {
  const sample = currentSample();
  window.hearsayPortrait('sample-portrait', sample?.title);
  $('sample-use').disabled = !sample;
  $('sample-download').hidden = !sample;
  if (!sample) {
    $('sample-player').removeAttribute('src'); $('sample-player').load();
    $('sample-details').textContent = 'No recordings match your search.'; return;
  }
  $('sample-player').src = sample.file;
  $('sample-download').href = sample.file;
  $('sample-details').replaceChildren(document.createTextNode(`${sample.title} · ${sample.label === 'real' ? 'Real' : 'Synthetic'} · ${sample.duration.toFixed(2)} seconds · ${sample.collection} · `));
  const source = node('a', 'Dataset source'); source.href = sample.source;
  source.target = '_blank'; source.rel = 'noopener noreferrer'; $('sample-details').append(source);
}
function filterSamples() {
  const query = $('sample-search').value.trim().toLowerCase();
  visibleSamples = sampleCatalog.filter(s => `${s.title} ${s.label} ${s.collection} ${s.id}`.toLowerCase().includes(query));
  $('sample-select').replaceChildren(...visibleSamples.map(s => {
    const option = node('option', `${s.title} — ${s.label === 'real' ? 'Real' : 'Synthetic'} (${s.id})`); option.value = s.id; return option;
  }));
  $('sample-select').disabled = !visibleSamples.length; showSample();
}
$('sample-search').addEventListener('input', filterSamples);
$('sample-select').addEventListener('change', showSample);
$('sample-use').addEventListener('click', async () => {
  const sample = currentSample(); if (!sample) return;
  $('sample-use').disabled = true; $('sample-status').textContent = 'Loading recording…';
  try {
    const response = await fetch(sample.file);
    if (!response.ok) throw new Error('Could not load recording. Please retry.');
    const transfer = new DataTransfer();
    transfer.items.add(new File([await response.blob()], `${sample.title.replace(/[^a-z0-9 -]/gi, '')}-${sample.label}-${sample.id}.mp3`, {type:'audio/mpeg'}));
    $('audio').files = transfer.files; $('expected').value = sample.label;
    $('sample-status').textContent = `${sample.title} (${sample.label}) selected. Enter your access code, choose a mode, then click Run.`;
    $('form').scrollIntoView({behavior:'smooth', block:'start'});
  } catch (error) { $('sample-status').textContent = error.message; }
  finally { $('sample-use').disabled = !currentSample(); }
});
fetch('samples/library/catalog.json').then(r => { if (!r.ok) throw new Error(); return r.json(); })
  .then(data => { sampleCatalog = data; filterSamples(); })
  .catch(() => { $('sample-status').textContent = 'Could not load the library. Reload to retry; file uploads are still available.'; });
