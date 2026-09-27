// Author: Alex Picon <alexnpc@me.com>
(() => {
  const el = id => document.getElementById(id);
  let pairs = [], catalog = new Map(), deck = [], position = 0;
  let clips = [], answered = false, correct = 0, total = 0, loading = false;
  let heard = new Set();
  const feedback = text => { el('game-feedback').textContent = text; };
  function shuffled(items) {
    const copy = [...items];
    for (let i = copy.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [copy[i], copy[j]] = [copy[j], copy[i]];
    }
    return copy;
  }
  function makeDeck() {
    deck = shuffled(pairs.filter(p => !el('game-speaker').value || p.speaker === el('game-speaker').value));
    position = 0;
  }
  function tally() {
    el('game-score').textContent = `${correct} correct / ${total} answered${total ? ` · ${Math.round(correct / total * 100)}% this session` : ''}`;
  }
  function round() {
    if (!pairs.length || loading) return;
    if (position >= deck.length) makeDeck();
    const pair = deck[position++]; if (!pair) return;
    clips = shuffled([catalog.get(pair.real), catalog.get(pair.fake)]);
    answered = false; heard = new Set();
    el('game-name').textContent = pair.speaker;
    el('game-progress').textContent = `Pair ${position} of ${deck.length} in this shuffled deck · A/B order is randomized`;
    ['a','b'].forEach((side, index) => {
      const audio = el(`game-${side}`); audio.pause(); audio.src = clips[index].file;
      el(`guess-${side}`).disabled = true;
      el(`answer-${side}`).textContent = '';
      el(`analyze-${side}`).hidden = true;
      el(`game-${side}`).closest('article').classList.remove('revealed-real','revealed-synthetic');
    });
    el('game-next').disabled = true;
    feedback('Play both recordings to unlock your guess. You can replay them as often as you like.');
  }
  function guess(index) {
    if (answered || heard.size !== 2 || !clips.length) return;
    answered = true; total++;
    const won = clips[index].label === 'synthetic'; if (won) correct++;
    tally();
    ['a','b'].forEach((side, i) => {
      el(`guess-${side}`).disabled = true;
      el(`answer-${side}`).textContent = `${clips[i].label === 'real' ? 'Genuine recording' : 'Synthetic recording'} · Dataset label`;
      el(`game-${side}`).closest('article').classList.add(`revealed-${clips[i].label}`);
      el(`analyze-${side}`).hidden = false;
    });
    feedback(`${won ? 'You got it.' : 'That one fooled you.'} Recording ${clips[0].label === 'synthetic' ? 'A' : 'B'} is labeled synthetic. Different recording conditions can be misleading; this reveal does not identify which acoustic cue caused your choice. Try HEARSAY on either clip below.`);
    el('game-next').disabled = false;
  }
  async function analyze(index) {
    if (!answered || loading) return;
    const clip = clips[index]; loading = true;
    ['analyze-a','analyze-b','game-next','game-speaker','game-reset'].forEach(id => { el(id).disabled = true; });
    try {
      const response = await fetch(clip.file);
      if (!response.ok) throw new Error('Recording could not be loaded. Try again.');
      const transfer = new DataTransfer();
      transfer.items.add(new File([await response.blob()], `${clip.title.replace(/[^a-z0-9 -]/gi,'')}-${clip.label}-${clip.id}.mp3`, {type:'audio/mpeg'}));
      el('audio').files = transfer.files; el('expected').value = clip.label;
      el('message').textContent = `Game recording ${index === 0 ? 'A' : 'B'} selected (${clip.title}, ${clip.label}). Enter your access code, select a mode and run a fresh analysis.`;
      el('form').scrollIntoView({behavior:'smooth',block:'start'});
      el('key').focus({preventScroll:true});
    } catch (error) { feedback(error.message); }
    finally {
      loading = false;
      ['analyze-a','analyze-b','game-next','game-speaker','game-reset'].forEach(id => { el(id).disabled = false; });
    }
  }
  ['a','b'].forEach((side,index) => {
    el(`game-${side}`).addEventListener('play', () => {
      el(`game-${index === 0 ? 'b' : 'a'}`).pause();
      heard.add(side);
      if (!answered && heard.size === 2) {
        el('guess-a').disabled = false; el('guess-b').disabled = false;
        feedback('Which recording sounds synthetic? Make your choice, or listen again.');
      }
    });
    el(`game-${side}`).addEventListener('error', () => feedback('Audio could not load. Reload the page to retry.'));
    el(`guess-${side}`).addEventListener('click', () => guess(index));
    el(`analyze-${side}`).addEventListener('click', () => analyze(index));
  });
  el('game-next').addEventListener('click', round);
  el('game-speaker').addEventListener('change', () => { makeDeck(); round(); });
  el('game-reset').addEventListener('click', () => { correct = 0; total = 0; tally(); makeDeck(); round(); });
  Promise.all(['samples/library/pairs.json','samples/library/catalog.json'].map(async url => {
    const response = await fetch(url); if (!response.ok) throw new Error(); return response.json();
  })).then(([sourcePairs, samples]) => {
    catalog = new Map(samples.map(s => [s.id,s]));
    pairs = sourcePairs.filter(p => catalog.get(p.real)?.label === 'real' && catalog.get(p.fake)?.label === 'synthetic');
    if (!pairs.length) throw new Error();
    [...new Set(pairs.map(p => p.speaker))].sort().forEach(speaker => {
      const option = document.createElement('option'); option.value = speaker; option.textContent = speaker;
      el('game-speaker').append(option);
    });
    makeDeck(); round();
  }).catch(() => { feedback('The game could not load. Reload to retry; the analysis form below is still available.'); });
  function illustrate() {
    const prevalence = Number(el('prevalence').value);
    const fake = prevalence * 10, caught = Math.round(fake * .9), falseAlarms = Math.round((1000 - fake) * .05);
    el('prevalence-value').textContent = `${prevalence}%`;
    el('base-rate-result').textContent = `Of 1,000 recordings: ${fake} are synthetic. About ${caught} fakes are caught and ${falseAlarms} genuine recordings are wrongly flagged. Roughly ${Math.round(caught / (caught + falseAlarms) * 100)}% of flagged recordings are actually synthetic. About ${fake - caught} fakes are missed.`;
  }
  el('prevalence').addEventListener('input', illustrate); illustrate();
})();
