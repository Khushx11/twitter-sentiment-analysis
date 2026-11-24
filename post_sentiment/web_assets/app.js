
const $ = id => document.getElementById(id);
let inputVersion = 0;
function updateCount() {
  inputVersion += 1;
  $('count').textContent = $('post').value.length + ' / 1,000';
  $('result').hidden = true;
  $('empty').hidden = false;
  $('status').textContent = '';
  $('status').classList.remove('error');
}
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
  const button = event.submitter;
  const version = inputVersion;
  if (button) button.disabled = true;
  $('status').textContent = 'Analysing…';
  $('status').classList.remove('error');
  $('result').hidden = true;
  $('empty').hidden = true;
  try {
    const response = await fetch('/api/predict', {method:'POST', body:$('post').value});
    const value = await response.json();
    if (version !== inputVersion) return;
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
    if (version !== inputVersion) return;
    $('status').textContent = error.message;
    $('status').classList.add('error');
    $('empty').hidden = false;
  } finally {
    if (button) button.disabled = false;
  }
});
function showMetrics(values) {
  $('metrics').replaceChildren(...values.map(([name, score]) => {
    const card = label('div', '');
    card.className = 'metric';
    card.append(label('strong', (score * 100).toFixed(1) + '%'), label('small', name));
    return card;
  }));
  $('metrics').hidden = false;
}
function showProtocol(entries) {
  $('protocol').replaceChildren(...entries.map(([name, value]) => {
    const row = label('div', '');
    row.append(label('dt', name), label('dd', value));
    return row;
  }));
}
function drawMatrix(report) {
  const names = report.class_order;
  const result = report.logistic_regression;
  const header = label('tr', '');
  for (const name of ['Actual / predicted', ...names, 'Recall', 'Support']) {
    const cell = label('th', name);
    cell.scope = 'col';
    header.append(cell);
  }
  const head = label('thead', '');
  head.append(header);
  const body = label('tbody', '');
  names.forEach((name, index) => {
    const row = label('tr', '');
    const title = label('th', name);
    title.scope = 'row';
    const support = result.class_metrics[name].support;
    const cells = result.confusion_matrix[index].map(count => {
      const cell = label('td', count.toLocaleString());
      const share = support ? count / support : 0;
      cell.className = 'heat level-' + Math.round(5 * share);
      cell.title = (share * 100).toFixed(1) + '% of actual ' + name + ' tweets';
      return cell;
    });
    row.append(title, ...cells, label('td', (100 * result.class_metrics[name].recall).toFixed(1) + '%'),
      label('td', support.toLocaleString()));
    body.append(row);
  });
  $('matrix').replaceChildren(head, body);
  $('diagnostics').hidden = false;
}
async function loadSession() {
  try {
    const response = await fetch('/api/session');
    if (!response.ok) throw new Error('Model session unavailable.');
    const value = await response.json();
    $('mode').textContent = value.mode;
    const result = value.evaluation;
    if (result?.dataset === 'TweetEval sentiment') {
      const test = result.split_sizes.test.toLocaleString();
      const overlap = result.overlap_with_train.test;
      $('session').textContent = `TweetEval official test split · ${test} posts. Validation macro recall selected the regularisation setting. ${overlap} test posts match training text after normalisation.`;
      $('dataset-link').href = result.source;
      $('dataset-link').hidden = false;
      showMetrics([
        ['Test macro recall', result.logistic_regression.macro_recall],
        ...(result.linear_svm ? [['Linear SVM macro recall', result.linear_svm.macro_recall]] : []),
        ['Majority macro recall', result.majority_baseline.macro_recall],
        ['Test macro-F1', result.logistic_regression.macro_f1],
      ]);
      drawMatrix(result);
      showProtocol([
        ['Training tweets', result.split_sizes.train.toLocaleString()],
        ['Validation tweets', result.split_sizes.val.toLocaleString()],
        ['Test tweets', test],
        ...(result.test_vocabulary_coverage ? [[
          'Test tweets declined by the vocabulary check',
          result.test_vocabulary_coverage.without_features.toLocaleString(),
        ]] : []),
        ['Features', 'Word + character TF-IDF'],
        ['Estimator', 'Logistic regression'],
        ['Logistic regression C', result.selected_C],
        ...(result.svm_selected_C === undefined ? [] : [['Linear SVM C', result.svm_selected_C]]),
      ]);
    } else if (result) {
      $('session').textContent = 'Held-out results from the dataset used to train this local model.';
      showMetrics([
        ['Model macro-F1', result.logistic_regression.macro_f1],
        ['Majority macro-F1', result.majority_baseline.macro_f1],
        ['Test accuracy', result.logistic_regression.accuracy],
      ]);
      drawMatrix(result);
      showProtocol([
        ['Training examples', Object.values(result.train_counts).reduce((sum, n) => sum + n, 0)],
        ['Test examples', Object.values(result.test_counts).reduce((sum, n) => sum + n, 0)],
        ['Features', 'Word + character TF-IDF'],
        ['Estimator', 'Logistic regression'],
      ]);
    } else if (value.held_out) {
      $('session').textContent = 'Synthetic smoke test on eight posts. Load the TweetEval model to see the benchmark evaluation.';
    } else {
      $('session').textContent = 'This local model has no embedded evaluation record.';
    }
  } catch (error) { $('session').textContent = error.message; }
}
loadSession();
