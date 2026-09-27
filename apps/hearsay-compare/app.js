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
