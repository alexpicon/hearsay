// Author: Alex Picon <alexnpc@me.com>
(() => {
  const catalog = fetch('portraits/catalog.json').then(r => {
    if (!r.ok) throw new Error('Portrait catalog unavailable');
    return r.json();
  }).catch(() => ({}));
  window.hearsayPortrait = async (id, speaker) => {
    const target = document.getElementById(id);
    target.dataset.speaker = speaker || '';
    target.hidden = true;
    target.replaceChildren();
    const portraits = await catalog;
    if (target.dataset.speaker !== (speaker || '')) return;
    const portrait = portraits[speaker];
    if (!portrait) return;
    const image = document.createElement('img');
    image.className = 'speaker-photo'; image.width = 72; image.height = 72;
    image.alt = `Portrait of ${portrait.person}`;
    image.src = portrait.file; image.decoding = 'async';
    image.addEventListener('error', () => { target.hidden = true; });
    const credit = document.createElement('a');
    credit.href = 'portraits/#' + portrait.file.split('/').pop().replace(/\.[^.]+$/, '');
    credit.textContent = 'Photo credit'; credit.className = 'portrait-credit';
    credit.setAttribute('aria-label', `Photo credit for ${portrait.person}`);
    target.append(image, credit); target.hidden = false;
  };
})();
