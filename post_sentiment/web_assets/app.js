
const $ = id => document.getElementById(id);
const updateCount = () => { $('count').textContent = $('post').value.length + ' / 1,000'; };
$('post').addEventListener('input', updateCount);
document.querySelectorAll('[data-text]').forEach(button => button.addEventListener('click', () => {
  $('post').value = button.dataset.text;
  updateCount();
  $('post').focus();
}));
function label(tag, text) {
  const element = document.createElement(tag);
  element.textContent = text;
  return element;
}
$('predict').addEventListener('submit', async event => {
  event.preventDefault();
  $('status').textContent = 'Analysing…';
  $('status').classList.remove('error');
  $('result').hidden = true;
  $('empty').hidden = true;
  try {
    const response = await fetch('/api/predict', {method:'POST', body:$('post').value});
    const value = await response.json();
    if (!response.ok) throw new Error(value.error);
    $('label').textContent = value.label;
    $('scores').replaceChildren(...Object.entries(value.model_scores).map(([name, score]) => {
      const row = label('div', '');
      row.className = 'score';
      const bar = document.createElement('progress');
      bar.max = 1;
      bar.value = score;
      bar.setAttribute('aria-label', name + ' model score');
      row.append(label('span', name), bar, label('span', (score * 100).toFixed(1) + '%'));
      return row;
    }));
    $('result').hidden = false;
    $('status').textContent = 'Analysis complete.';
  } catch (error) {
    $('status').textContent = error.message;
    $('status').classList.add('error');
    $('empty').hidden = false;
  }
});
async function loadSession() {
  try {
    const response = await fetch('/api/session');
    if (!response.ok) throw new Error('Model session unavailable.');
    const value = await response.json();
    $('mode').textContent = value.mode;
    $('session').textContent = value.held_out
      ? `Scores below use ${value.held_out} held-out synthetic posts. They show that evaluation runs, not how the model performs on real posts.`
      : 'Using the model supplied when this local session started.';
    if (value.held_out) {
      $('metrics').replaceChildren(...[['Model macro-F1', value.model_f1], ['Majority baseline macro-F1', value.baseline_f1]].map(([name, score]) => {
        const card = label('div', '');
        card.className = 'metric';
        card.append(label('strong', score.toFixed(3)), label('small', name));
        return card;
      }));
      $('metrics').hidden = false;
    }
  } catch (error) { $('session').textContent = error.message; }
}
loadSession();
