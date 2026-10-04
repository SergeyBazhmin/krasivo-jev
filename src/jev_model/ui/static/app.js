const $ = (id) => document.getElementById(id);
let runs = [];
let loaded = null;
let busy = false;
let imported = null;

async function api(path, method = 'GET', body) {
  const response = await fetch(`/api/${path}`, {
    method,
    headers: body ? { 'Content-Type': 'application/json' } : {},
    body: body ? JSON.stringify(body) : undefined,
  });
  const data = await response.json();
  if (!response.ok) {
    const detail = Array.isArray(data.detail)
      ? data.detail.map((e) => `${e.loc.slice(1).join('.')}: ${e.msg}`).join('; ')
      : data.detail;
    throw new Error(detail || `HTTP ${response.status}`);
  }
  return data;
}

function showError(error) {
  $('error').textContent = error.message;
  $('error').hidden = false;
}

function options() {
  return $('options').value.split('\n').map((text) => text.trim()).filter(Boolean);
}

function updateControls() {
  const selected = runs.find((run) => run.run === $('run').value);
  $('load').disabled = busy || !selected?.stages.length;
  $('unload').disabled = busy || !loaded;
  $('predict').disabled = busy || !loaded || !$('question').value.trim() || options().length < 2;
  $('load-sample').disabled = busy || !$('dataset').value;
  for (const id of ['run', 'stage', 'device', 'dataset', 'partition', 'sample-index', 'sample-file',
                    'state', 'question', 'options']) $(id).disabled = busy;
}

function setStatus(info) {
  loaded = info;
  $('model-status').textContent = info ? 'Загружена' : 'Не загружена';
  $('model-status').classList.toggle('loaded', !!info);
  $('model-detail').textContent = info
    ? `${info.run} · ${info.stage} · ${info.device} · загрузка ${(info.load_ms / 1000).toFixed(2)} с`
    : 'Выберите run с обученными весами.';
  updateControls();
}

function clearResult() {
  $('timing').textContent = '';
  $('result').replaceChildren();
  const placeholder = document.createElement('p');
  placeholder.className = 'empty hint';
  placeholder.textContent = 'Отправьте пример, чтобы получить результат.';
  $('result').append(placeholder);
}

async function action(buttonId, message, work) {
  if (busy) return;
  busy = true;
  $('error').hidden = true;
  const button = $(buttonId);
  const text = button.textContent;
  button.textContent = message;
  updateControls();
  try { await work(); } catch (error) { showError(error); }
  finally { busy = false; button.textContent = text; updateControls(); }
}

function selectRun() {
  const run = runs.find((item) => item.run === $('run').value);
  $('stage').replaceChildren(new Option('Последняя', ''));
  for (const stage of run?.stages || []) $('stage').add(new Option(stage, stage));
  updateControls();
}

async function refreshDatasets() {
  const { datasets } = await api('datasets');
  $('dataset').replaceChildren();
  for (const name of datasets) $('dataset').add(new Option(name, name));
  $('dataset-detail').textContent = datasets.length ? '' : 'Нет подготовленных датасетов. Можно ввести пример вручную.';
  updateControls();
}

function fillSample(sample) {
  if (!sample || typeof sample.question !== 'string' || !sample.question.trim() ||
      (sample.state !== undefined && typeof sample.state !== 'string') ||
      !Array.isArray(sample.options) || sample.options.length < 2) {
    throw new Error('Нужен JSON с question, options (минимум 2 варианта) и необязательным state.');
  }
  const normalized = sample.options.map((option, index) => {
    if (typeof option === 'string') return { id: String(index), text: option };
    if (option && typeof option.id === 'string' && typeof option.text === 'string') return option;
    throw new Error('options: строки или объекты {"id": "...", "text": "..."}.');
  });
  if (normalized.some((option) => !option.id.trim() || !option.text.trim() || /[\r\n]/.test(option.text)) ||
      new Set(normalized.map((option) => option.id)).size !== normalized.length) {
    throw new Error('Каждый вариант должен иметь уникальный id и непустой текст в одну строку.');
  }
  $('state').value = sample.state || '';
  $('question').value = sample.question;
  $('options').value = normalized.map((option) => option.text).join('\n');
  imported = { options: normalized, label: sample.label };
  clearResult();
  updateControls();
}

function node(tag, className, text) {
  const element = document.createElement(tag);
  element.className = className;
  if (text !== undefined) element.textContent = text;
  return element;
}

function renderResult(result, label) {
  $('result').replaceChildren();
  $('timing').textContent = `${result.elapsed_ms.toFixed(1)} мс`;
  const winner = node('div', 'winner');
  winner.append(node('span', 'hint', 'Выбранный ответ'), node('span', 'winner-text', result.prediction.text),
    node('span', 'winner-prob', `${(result.prediction.probability * 100).toFixed(1)}%`));
  $('result').append(winner);
  for (const option of result.options) {
    const row = node('div', 'prob-row');
    const heading = node('div', 'prob-label');
    heading.append(node('span', '', option.text), node('span', '', `${(option.probability * 100).toFixed(2)}%`));
    const track = node('div', 'track');
    const bar = node('div', option.id === result.prediction.id ? 'bar best' : 'bar');
    bar.style.width = `${option.probability * 100}%`;
    track.append(bar);
    row.append(heading, track);
    $('result').append(row);
  }
  if (Array.isArray(label) && label.length === result.options.length &&
      label.every((value) => typeof value === 'number' && Number.isFinite(value) && value >= 0) &&
      label.some((value) => value > 0)) {
    const target = label.indexOf(Math.max(...label));
    const expected = result.options[target];
    const text = expected.id === result.prediction.id
      ? `Совпадает с меткой: ${expected.text}` : `Ожидаемый ответ по метке: ${expected.text}`;
    $('result').append(node('p', 'label-result', text));
  }
}

$('run').addEventListener('change', selectRun);
$('load').addEventListener('click', () => action('load', 'Загрузка…', async () => {
  clearResult();
  try {
    const data = await api('model', 'POST', {
      run: $('run').value, stage: $('stage').value || null, device: $('device').value,
    });
    setStatus(data.loaded);
  } catch (error) {
    setStatus((await api('model')).loaded);
    throw error;
  }
  await refreshDatasets();
}));
$('unload').addEventListener('click', () => action('unload', 'Выгрузка…', async () => {
  setStatus((await api('model', 'DELETE')).loaded);
  clearResult();
  await refreshDatasets();
}));
$('sample-form').addEventListener('submit', (event) => {
  event.preventDefault();
  if (!loaded) return;
  action('predict', 'Обработка…', async () => {
    clearResult();
    const sampleOptions = imported?.options || options();
    const result = await api('predict', 'POST', {
      state: $('state').value, question: $('question').value, options: sampleOptions, model_id: loaded.id,
    });
    renderResult(result, imported?.label);
  });
});
for (const id of ['state', 'question', 'options']) $(id).addEventListener('input', () => {
  imported = null;
  clearResult();
  updateControls();
});
$('sample-file').addEventListener('change', async () => {
  $('error').hidden = true;
  try {
    const file = $('sample-file').files[0];
    if (file) fillSample(JSON.parse(await file.text()));
  } catch (error) { showError(error); }
  finally { $('sample-file').value = ''; }
});
$('load-sample').addEventListener('click', () => action('load-sample', 'Загрузка…', async () => {
  const query = new URLSearchParams({ dataset: $('dataset').value, partition: $('partition').value,
    index: $('sample-index').value });
  const data = await api(`sample?${query}`);
  fillSample(data.sample);
  $('dataset-detail').textContent = `Строка ${data.index} · всего ${data.count}`;
}));

async function initialize() {
  try {
    const [listing, status] = await Promise.all([api('runs'), api('model')]);
    runs = listing.runs;
    $('run').replaceChildren();
    for (const run of runs) $('run').add(new Option(`${run.run} · ${run.model}`, run.run));
    if (!runs.length) $('run').add(new Option('Нет сохранённых моделей', ''));
    if (status.loaded) {
      $('run').value = status.loaded.run;
      $('device').value = status.loaded.device;
    }
    selectRun();
    if (status.loaded) $('stage').value = status.loaded.stage;
    setStatus(status.loaded);
    if (!runs.length) $('model-detail').textContent = `Нет runs в ${listing.runs_dir}. Сначала обучите модель через jev-model train.`;
    await refreshDatasets();
  } catch (error) { showError(error); updateControls(); }
}
initialize();
