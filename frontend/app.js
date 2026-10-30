/* OpenGenjutsu front-end — vanilla ES2020, no build step.
 *
 * Works in two modes:
 *   1. served by the backend  -> same-origin requests, base = ""
 *   2. opened from disk/static host -> requests go to the configured backend URL
 *
 * Every media URL gets ?token= appended so <video src> works when the backend
 * is protected by a shared secret.
 */

(() => {
  'use strict';

  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

  const LS = {
    base: 'og.base',
    token: 'og.token',
    form: 'og.form',
  };

  const STAGE_LABELS = {
    queued: 'Queued',
    starting: 'Starting',
    'preparing sources': 'Preparing sources',
    'staging inputs': 'Sending to GPU',
    'building workflow': 'Building workflow',
    'loading cached graph': 'Loading cached graph',
    generating: 'Generating',
    sampling: 'Generating',
    'retrieving render': 'Retrieving render',
    finalising: 'Finalising',
    done: 'Done',
    canceled: 'Canceled',
    failed: 'Failed',
  };

  const urlParams = new URLSearchParams(window.location.search);
  const paramToken = urlParams.get('token') || (window.location.hash.startsWith('#token=') ? window.location.hash.slice(7) : '');
  if (paramToken) {
    localStorage.setItem(LS.token, paramToken);
    try {
      const cleanUrl = window.location.origin + window.location.pathname;
      window.history.replaceState({}, document.title, cleanUrl);
    } catch (_) {}
  }

  const state = {
    base: localStorage.getItem(LS.base) || '',
    token: paramToken || localStorage.getItem(LS.token) || '',
    engines: [],
    defaultEngine: null,
    engine: null,
    health: null,
    files: { motion: null, character: null },
    job: null,
    pollTimer: null,
    view: 'result',
    submitting: false,
    touched: new Set(),
  };

  const el = {};

  /* ------------------------------------------------------------------ utils */

  const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  function fmtBytes(n) {
    if (!n) return '—';
    const units = ['B', 'KB', 'MB', 'GB'];
    let i = 0;
    let v = n;
    while (v >= 1024 && i < units.length - 1) { v /= 1024; i++; }
    return `${v.toFixed(v < 10 ? 1 : 0)} ${units[i]}`;
  }

  function fmtTime(s) {
    if (s == null || Number.isNaN(s)) return '—';
    if (s < 60) return `${s.toFixed(1)}s`;
    const m = Math.floor(s / 60);
    return `${m}m ${Math.round(s - m * 60)}s`;
  }

  function toast(message, kind = 'info', ttl = 5200) {
    const node = document.createElement('div');
    node.className = 'toast';
    node.dataset.kind = kind;
    node.textContent = message;
    el.toasts.append(node);
    setTimeout(() => {
      node.style.transition = 'opacity .25s ease, transform .25s ease';
      node.style.opacity = '0';
      node.style.transform = 'translateY(6px)';
      setTimeout(() => node.remove(), 300);
    }, ttl);
  }

  function normalizeBase(value) {
    const raw = (value || '').trim().replace(/\/+$/, '');
    if (!raw) return '';
    return /^https?:\/\//i.test(raw) ? raw : `http://${raw}`;
  }

  /** Absolute URL for a path, honouring the configured backend. */
  function absUrl(path) {
    if (!path) return '';
    if (/^https?:\/\//i.test(path)) return path;
    const origin = state.base || (location.protocol === 'file:' ? '' : location.origin);
    return `${origin}${path}`;
  }

  function mediaUrl(path, extra = {}) {
    if (!path) return '';
    const url = new URL(absUrl(path), 'http://placeholder.local');
    if (state.token) url.searchParams.set('token', state.token);
    for (const [k, v] of Object.entries(extra)) url.searchParams.set(k, v);
    // Same-origin URLs must stay relative so they work behind tunnels/paths.
    const origin = state.base || (location.protocol === 'file:' ? '' : location.origin);
    return origin ? url.toString() : `${url.pathname}${url.search}`;
  }

  function headers(extra = {}) {
    const h = { ...extra };
    if (state.token) h['X-Auth-Token'] = state.token;
    return h;
  }

  async function api(path, options = {}) {
    const url = absUrl(path);
    const res = await fetch(url, { ...options, headers: headers(options.headers) });
    if (!res.ok) {
      let detail;
      try {
        const payload = await res.json();
        detail = payload.detail ?? payload;
      } catch {
        detail = res.statusText;
      }
      const error = new Error(
        typeof detail === 'string' ? detail : detail?.message || JSON.stringify(detail)
      );
      error.status = res.status;
      error.detail = detail;
      throw error;
    }
    return res.headers.get('content-type')?.includes('json') ? res.json() : res;
  }

  /* ------------------------------------------------------------------ status */

  function setPill(stateName, text) {
    el.pill.dataset.state = stateName;
    $('span', el.pill).textContent = text;
  }

  /** Trim vendor noise so the status pill stays readable: "GTX 1650 Ti". */
  function shortGpu(name) {
    return String(name || 'GPU')
      .replace(/^NVIDIA\s+/i, '')
      .replace(/^GeForce\s+/i, '')
      .replace(/\s+\(.*\)$/, '');
  }

  function engineById(id) {
    return state.engines.find((e) => e.id === id) || null;
  }

  /* ------------------------------------------------------------------ connect */

  async function refreshHealth({ quiet = false } = {}) {
    try {
      const health = await api('/api/health');
      state.health = health;
      const gpu = health.gpu?.available ? shortGpu(health.gpu.name) : 'No GPU';
      const busy = health.queue?.active;
      setPill(busy ? 'busy' : 'ok', busy ? 'Rendering' : 'Ready');
      el.pill.title = `${health.gpu?.available ? health.gpu.name : 'No CUDA device'} · v${health.version}`;

      if (el.gpuBadge) {
        if (health.gpu?.available) {
          const vramGb = Math.round((health.gpu.vram_total_mb || 0) / 1024);
          const isPro = (health.gpu.vram_total_mb || 0) >= 18000;
          el.gpuBadge.hidden = false;
          el.gpuBadge.textContent = `${gpu} · ${vramGb}GB${isPro ? ' (Pro)' : ''}`;
          el.gpuBadge.className = `gpu-badge${isPro ? ' gpu-badge--pro' : ''}`;
          el.gpuBadge.title = `${health.gpu.name} (${health.gpu.vram_total_mb} MB VRAM) · Driver: ${health.gpu.driver || 'N/A'}`;
        } else {
          el.gpuBadge.hidden = true;
        }
      }

      renderHealth(health);
      return health;
    } catch (error) {
      setPill('error', error.status === 401 ? 'Locked' : 'Offline');
      if (!quiet) {
        if (error.status === 401) {
          toast('Access token rejected — open settings and paste the token from the notebook.', 'error', 9000);
        } else {
          toast(`Cannot reach backend at ${state.base || location.origin}`, 'error', 7000);
        }
      }
      return null;
    }
  }

  function renderHealth(health) {
    if (!health) {
      el.health.innerHTML = '<div><span>Backend</span><b class="bad">unreachable</b></div>';
      return;
    }
    const rows = [];
    const ok = (v) => `<b class="${v ? 'ok' : 'bad'}">${v ? 'yes' : 'no'}</b>`;
    rows.push(`<div><span>Version</span><b>${health.version}</b></div>`);
    rows.push(`<div><span>ComfyUI</span>${ok(health.comfy?.reachable)}</div>`);
    rows.push(
      `<div><span>GPU</span><b>${
        health.gpu?.available
          ? `${health.gpu.name} · ${Math.round(health.gpu.vram_total_mb / 1024)} GB`
          : 'not visible'
      }</b></div>`
    );
    rows.push(
      `<div><span>Queue</span><b>${health.queue?.queued ?? 0} waiting${health.queue?.active ? ' · 1 running' : ''}</b></div>`
    );
    rows.push(
      `<div><span>Model staging</span><b>${health.comfy?.staging_mode || '—'}</b></div>`
    );
    rows.push(
      `<div><span>Disk free</span><b>${fmtBytes((health.storage?.free_mb || 0) * 1024 * 1024)}</b></div>`
    );
    rows.push(`<div><span>ffmpeg</span>${ok(health.ffmpeg?.ok)}</div>`);
    rows.push(
      `<div><span>Auth</span><b>${health.auth?.required ? 'token required' : 'open'}</b></div>`
    );
    el.health.innerHTML = rows.join('');
  }

  async function loadEngines() {
    const payload = await api('/api/engines');
    state.engines = payload.engines || [];
    state.defaultEngine = payload.default || 'mock';

    // Ready engines first, then unavailable ones, mock last (it is a test tool).
    const rank = (engine) => (engine.id === 'mock' ? 3 : engine.ready ? 0 : 1);
    const ordered = [...state.engines].sort((a, b) => rank(a) - rank(b));

    const previous = state.engine || localStorage.getItem('og.engine');
    el.engine.innerHTML = '';

    for (const engine of ordered) {
      const option = document.createElement('option');
      option.value = engine.id;
      const suffix = engine.ready ? '' : ' (unavailable)';
      option.textContent = `${engine.label}${suffix}`;
      option.title = [engine.description, ...(engine.problems || [])].join('\n\n');
      el.engine.append(option);
    }

    const previousEngine = previous && engineById(previous);
    const wanted = previousEngine?.ready ? previous : (state.defaultEngine || ordered.find((e) => e.ready)?.id || 'mock');
    state.engine = wanted;
    el.engine.value = wanted;
    applyEngine(wanted, { silent: true });
  }

  function applyEngine(engineId, { silent = false } = {}) {
    state.engine = engineId;
    localStorage.setItem('og.engine', engineId);
    const engine = engineById(engineId);
    if (!engine) return;

    const d = engine.defaults || {};
    const maxSeconds = Number(d.max_seconds || 8);

    el.seconds.max = String(clamp(maxSeconds, 1, 30));
    if (!state.touched.has('seconds')) {
      el.seconds.value = String(clamp(Number(d.frames || 49) / Number(d.fps || 16), 1, maxSeconds));
    }
    if (!state.touched.has('steps')) el.steps.value = d.steps ?? 16;
    if (!state.touched.has('cfg')) el.cfg.value = d.cfg ?? 4;
    if (!state.touched.has('fps')) el.fps.value = d.fps ?? 16;
    updateSeconds();

    const problems = engine.problems || [];
    const notes = [];
    if (engine.description) notes.push(engine.description);
    if (problems.length) notes.push(problems.join(' '));
    else if ((engine.warnings || []).length) notes.push(engine.warnings[0]);

    // Full detail (including install notes) stays in the option tooltip; the
    // panel keeps one short paragraph so the layout does not grow.
    el.engineNote.textContent = notes.join(' ');
    el.engineNote.classList.toggle('warn', problems.length > 0);

    if (!silent) updateGenerateState();
  }

  /* ------------------------------------------------------------------ inputs */

  function bindDrop(kind) {
    const drop = kind === 'motion' ? el.dropMotion : el.dropCharacter;
    const input = kind === 'motion' ? el.fileMotion : el.fileCharacter;

    drop.addEventListener('click', (event) => {
      if (event.target.closest('[data-clear]')) return;
      input.click();
    });
    drop.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        input.click();
      }
    });
    input.addEventListener('change', () => {
      if (input.files?.[0]) setFile(kind, input.files[0]);
    });

    ['dragenter', 'dragover'].forEach((type) =>
      drop.addEventListener(type, (event) => {
        event.preventDefault();
        drop.classList.add('is-drag');
      })
    );
    ['dragleave', 'dragend'].forEach((type) =>
      drop.addEventListener(type, () => drop.classList.remove('is-drag'))
    );
    drop.addEventListener('drop', (event) => {
      event.preventDefault();
      drop.classList.remove('is-drag');
      const file = event.dataTransfer?.files?.[0];
      if (file) setFile(kind, file);
    });

    const clear = $('[data-clear]', drop);
    clear.addEventListener('click', (event) => {
      event.stopPropagation();
      clearFile(kind);
    });
  }

  function clearFile(kind) {
    state.files[kind] = null;
    if (kind === 'motion') {
      el.previewMotion.hidden = true;
      el.previewMotion.removeAttribute('src');
      el.metaMotion.hidden = true;
      el.fileMotion.value = '';
      el.dropMotion.classList.remove('has-file');
      $('.drop__empty', el.dropMotion).hidden = false;
      $('[data-clear="motion"]').hidden = true;
      el.audioNote.textContent = 'Re-attached to the finished render.';
    } else {
      el.previewCharacter.hidden = true;
      el.previewCharacter.removeAttribute('src');
      el.metaCharacter.hidden = true;
      el.fileCharacter.value = '';
      el.dropCharacter.classList.remove('has-file');
      $('.drop__empty', el.dropCharacter).hidden = false;
      $('[data-clear="character"]').hidden = true;
    }
    updateGenerateState();
  }

  function setFile(kind, file) {
    const isVideo = file.type.startsWith('video/') || /\.(mp4|mov|webm|mkv|m4v)$/i.test(file.name);
    if (kind === 'motion' && !isVideo) {
      toast('The motion source must be a video file.', 'error');
      return;
    }
    if (kind === 'character' && !file.type.startsWith('image/')) {
      toast('The character reference must be an image.', 'error');
      return;
    }

    state.files[kind] = file;
    const drop = kind === 'motion' ? el.dropMotion : el.dropCharacter;
    drop.classList.add('has-file');
    $('.drop__empty', drop).hidden = true;
    $(`[data-clear="${kind}"]`).hidden = false;

    if (kind === 'motion') {
      const video = el.previewMotion;
      video.hidden = false;
      video.src = URL.createObjectURL(file);
      video.muted = true;
      video.play().catch(() => {});
      video.onloadedmetadata = () => {
        const dur = video.duration;
        const w = video.videoWidth;
        const h = video.videoHeight;
        el.metaMotion.hidden = false;
        el.metaMotion.textContent = `${w}×${h} · ${dur.toFixed(2)}s · ${fmtBytes(file.size)}`;
        autoSeconds(dur);
        autoResolution(w, h);
        detectAudio(video, file);
      };
    } else {
      const img = el.previewCharacter;
      img.hidden = false;
      img.onload = () => {
        el.metaCharacter.hidden = false;
        el.metaCharacter.textContent = `${img.naturalWidth}×${img.naturalHeight} · ${fmtBytes(file.size)}`;
      };
      img.src = URL.createObjectURL(file);
    }
    updateGenerateState();
  }

  function autoSeconds(sourceDuration) {
    const max = Number(el.seconds.max) || 8;
    if (state.touched.has('seconds')) return;
    const dur = clamp(sourceDuration, 1, max);
    const stepped = Math.round(dur * 2) / 2;
    el.seconds.value = String(clamp(stepped, 1, max));
    updateSeconds();
  }

  function autoResolution(sourceWidth, sourceHeight) {
    if (!sourceWidth || !sourceHeight) return;
    if (state.touched.has('resolution')) return;

    const aspect = sourceWidth / sourceHeight;
    const isPro = (state.health?.gpu?.vram_total_mb || 0) >= 18000;
    let target = isPro ? '1280x720' : '832x480';
    let label = 'Landscape';

    if (aspect <= 0.8) {
      target = isPro ? '720x1280' : '480x832';
      label = 'Portrait';
    } else if (aspect >= 0.85 && aspect <= 1.15) {
      target = isPro ? '640x640' : '512x512';
      label = 'Square';
    }

    const opt = $$('option', el.resolution).find((o) => o.value === target);
    if (opt) {
      el.resolution.value = target;
      toast(`Auto-matched ${label} resolution (${target})`);
    }
  }

  /** Best-effort audio detection so the toggle can warn before submitting. */
  function detectAudio(video, file) {
    // Chrome/Firefox expose partial hints only once media is decoded.
    video.addEventListener(
      'loadeddata',
      () => {
        const decoded = video.webkitAudioDecodedByteCount;
        const tracks = video.audioTracks?.length;
        const hasAudio = (typeof decoded === 'number' && decoded > 0) || (tracks ?? 0) > 0;
        if (hasAudio) {
          el.audioNote.textContent = 'The motion clip has sound — it will be re-attached after rendering.';
        } else if (typeof decoded === 'number' || tracks != null) {
          el.audioNote.textContent = 'No audio detected in this clip — the render will be silent.';
        }
      },
      { once: true }
    );
    el.audioNote.textContent = `Sound check for ${file.name} …`;
  }

  /** Draw the filled portion of a range input without a second element. */
  function paintRange(input) {
    if (!input) return;
    const min = Number(input.min || 0);
    const max = Number(input.max || 100);
    const value = Number(input.value);
    const pct = max > min ? ((value - min) / (max - min)) * 100 : 0;
    input.style.setProperty('--pct', `${clamp(pct, 0, 100)}%`);
  }

  function updateSeconds() {
    const seconds = Number(el.seconds.value);
    const fps = Number(el.fps.value) || 16;
    // Wan needs 4n+1 frames; the backend rounds up, so show the same number.
    const raw = Math.max(5, Math.round(seconds * fps));
    const frames = Math.ceil((raw - 1) / 4) * 4 + 1;
    el.secondsOut.textContent = `${seconds.toFixed(1)}s · ${frames}f`;
    el.secondsOut.title = `${frames} frames at ${fps} fps`;
    paintRange(el.seconds);
  }

  function updateGenerateState() {
    const engine = engineById(state.engine);
    const ready = Boolean(engine?.ready);
    const haveFiles = Boolean(state.files.motion && state.files.character);
    const idle = !state.submitting && !isBusy();
    el.btnGenerate.disabled = !ready || !haveFiles || !idle;
    el.btnGenerate.title = !ready
      ? 'This engine is not ready — see the note under Engine.'
      : !haveFiles
      ? 'Add a motion clip and a character image first.'
      : 'Render (Ctrl + Enter)';
    el.btnCancel.hidden = !isBusy();
  }

  function isBusy() {
    return Boolean(state.job && ['queued', 'running'].includes(state.job.status));
  }

  /* ------------------------------------------------------------------ submit */

  function formPayload() {
    const [w, h] = (el.resolution.value || '832x480').split('x').map(Number);
    const keepAudio = el.keepAudio.checked;
    return {
      engine: state.engine,
      prompt: el.prompt.value.trim(),
      negative: el.negative.value.trim(),
      width: String(w),
      height: String(h),
      fps: el.fps.value,
      seconds: el.seconds.value,
      steps: el.steps.value,
      cfg: el.cfg.value,
      seed: el.seed.value,
      keep_audio: keepAudio ? 'true' : 'false',
      audio_volume: keepAudio ? el.audioVolume.value : '0',
    };
  }

  function saveForm() {
    const data = formPayload();
    delete data.keep_audio;
    localStorage.setItem(LS.form, JSON.stringify(data));
  }

  function restoreForm() {
    try {
      const saved = JSON.parse(localStorage.getItem(LS.form) || 'null');
      if (!saved) return;
      if (saved.prompt != null) el.prompt.value = saved.prompt;
      if (saved.negative != null) el.negative.value = saved.negative;
      if (saved.resolution) el.resolution.value = saved.resolution;
      for (const key of ['fps', 'steps', 'cfg', 'seed']) {
        if (saved[key] != null) {
          el[key].value = saved[key];
          state.touched.add(key);
        }
      }
      if (saved.seconds) {
        el.seconds.value = saved.seconds;
        state.touched.add('seconds');
        updateSeconds();
      }
    } catch {
      /* ignore corrupt storage */
    }
  }

  async function submit() {
    if (state.submitting || isBusy()) return;
    if (!state.files.motion || !state.files.character) {
      toast('Add both a motion video and a character image.', 'error');
      return;
    }

    state.submitting = true;
    updateGenerateState();
    el.btnGenerate.querySelector('span').textContent = 'Uploading…';

    try {
      const data = formPayload();
      const body = new FormData();
      body.append('motion', state.files.motion, state.files.motion.name);
      body.append('character', state.files.character, state.files.character.name);
      for (const [key, value] of Object.entries(data)) {
        if (value !== '' && value != null) body.append(key, value);
      }
      saveForm();

      const payload = await api('/api/jobs', { method: 'POST', body });
      const job = payload.job || payload;
      if (job.has_source_audio === false && el.keepAudio.checked) {
        toast('Heads up: the motion clip has no audio track, so this render will be silent.', 'error', 8000);
      }
      startPolling(job.id);
    } catch (error) {
      const problems = error.detail?.problems;
      toast(problems?.length ? `${error.message}: ${problems[0]}` : error.message, 'error', 9000);
      showEmpty();
    } finally {
      state.submitting = false;
      el.btnGenerate.querySelector('span').textContent = 'Generate';
      updateGenerateState();
    }
  }

  /* ------------------------------------------------------------------ polling */

  function startPolling(jobId) {
    stopPolling();
    state.job = { id: jobId, status: 'queued', progress: 0, stage: 'queued' };
    showProgress();
    updateGenerateState();
    const tick = async () => {
      try {
        const job = await api(`/api/jobs/${jobId}`);
        state.job = job;
        updateJob(job);
        if (['queued', 'running'].includes(job.status)) {
          state.pollTimer = setTimeout(tick, job.status === 'queued' ? 1600 : 900);
        } else {
          stopPolling();
          updateGenerateState();
          refreshHealth({ quiet: true });
          loadHistory();
        }
      } catch (error) {
        stopPolling();
        toast(`Lost contact with the backend: ${error.message}`, 'error', 9000);
        updateGenerateState();
      }
    };
    state.pollTimer = setTimeout(tick, 400);
  }

  function stopPolling() {
    if (state.pollTimer) clearTimeout(state.pollTimer);
    state.pollTimer = null;
  }

  function updateJob(job) {
    el.logs.textContent = (job.logs || []).join('\n');
    el.statEngine.textContent = engineById(job.engine)?.label || job.engine || '—';
    el.statElapsed.textContent = job.elapsed ? fmtTime(job.elapsed) : '—';

    if (job.status === 'queued') {
      showProgress();
      el.progressLabel.textContent = 'Queued';
      el.progressBar.style.width = '3%';
      el.progressSub.textContent = job.queue_position
        ? `position ${job.queue_position} in queue`
        : 'waiting for the GPU';
      setPill('busy', 'Queued');
      return;
    }

    if (job.status === 'running') {
      showProgress();
      setPill('busy', 'Rendering');
      const pct = clamp(job.progress || 0, 0, 1);
      el.progressLabel.textContent = STAGE_LABELS[job.stage] || job.stage || 'Working';
      el.progressBar.style.width = `${(pct * 100).toFixed(1)}%`;
      const bits = [`${Math.round(pct * 100)}%`];
      if (job.elapsed) bits.push(fmtTime(job.elapsed));
      if (pct > 0.08 && job.elapsed) {
        const eta = (job.elapsed / pct) * (1 - pct);
        if (eta > 1 && eta < 7200) bits.push(`~${fmtTime(eta)} left`);
      }
      el.progressSub.textContent = bits.join(' · ');
      el.statLength.textContent = job.params?.seconds ? `${job.params.seconds}s` : '—';
      return;
    }

    if (job.status === 'done') {
      setPill('ok', 'Ready');
      showResult(job);
      return;
    }

    if (job.status === 'error') {
      setPill('error', 'Failed');
      showError(job.error || 'Generation failed');
      el.logs.textContent = (job.logs || []).join('\n');
      toast(job.error || 'Generation failed', 'error', 10000);
      return;
    }

    if (job.status === 'canceled') {
      setPill('ok', 'Canceled');
      showEmpty('Canceled', 'The job was stopped.');
      toast('Generation canceled.', 'info');
    }
  }

  /* ------------------------------------------------------------------ views */

  function showEmpty(title = 'No render yet', sub = 'Add a motion clip and a character image, then generate.') {
    el.stageEmpty.hidden = false;
    $('p', el.stageEmpty).textContent = title;
    $('small', el.stageEmpty).textContent = sub;
    el.stageVideo.hidden = true;
    el.stageCompare.hidden = true;
    el.stageProgress.hidden = true;
    el.logsWrap.hidden = false;
    setView('result');
  }

  function showError(message) {
    el.stageEmpty.hidden = false;
    $('p', el.stageEmpty).textContent = 'Generation failed';
    $('small', el.stageEmpty).textContent = message;
    el.stageVideo.hidden = true;
    el.stageCompare.hidden = true;
    el.stageProgress.hidden = true;
    setView('result');
  }

  function showProgress() {
    el.stageEmpty.hidden = true;
    el.stageVideo.hidden = true;
    el.stageCompare.hidden = true;
    el.stageProgress.hidden = false;
    setView('result');
  }

  function showResult(job) {
    el.stageEmpty.hidden = true;
    el.stageProgress.hidden = true;
    const resultUrl = mediaUrl(job.urls?.result);
    const sourceUrl = mediaUrl(job.urls?.source);

    if (!resultUrl) {
      showError('The job finished but no file was produced.');
      return;
    }

    el.resultVideo.src = resultUrl;
    el.compareSource.src = sourceUrl || '';
    el.compareResult.src = resultUrl;

    const meta = job.meta || {};
    const bits = [];
    if (meta.width && meta.height) bits.push(`${meta.width}×${meta.height}`);
    if (meta.duration) bits.push(`${Number(meta.duration).toFixed(2)}s`);
    if (meta.fps) bits.push(`${meta.fps}fps`);
    bits.push(meta.has_audio ? `audio ${job.audio_applied ? 'original' : 'none'}` : 'silent');
    el.badgeMeta.textContent = bits.join(' · ');
    el.download.href = mediaUrl(job.urls?.result, { download: '1' });
    el.download.setAttribute('download', `ogenjutsu-${job.id}.mp4`);

    el.statLength.textContent = meta.duration ? `${Number(meta.duration).toFixed(2)}s` : '—';
    el.statAudio.textContent = meta.has_audio ? (job.audio_applied ? 'original' : 'ignored') : 'silent';

    setView(state.view === 'compare' ? 'compare' : 'result');
    updateGenerateState();
  }

  function setView(view) {
    const hasResult = Boolean(state.job?.urls?.result && state.job.status === 'done');
    if (view === 'compare' && !hasResult) view = 'result';
    state.view = view;
    $$('.tab').forEach((tab) => {
      const compare = tab.dataset.view === 'compare';
      tab.setAttribute('aria-selected', String(tab.dataset.view === view));
      tab.setAttribute('aria-disabled', String(compare && !hasResult));
    });
    if (!hasResult) return;
    el.stageEmpty.hidden = true;
    el.stageProgress.hidden = true;
    if (view === 'compare') {
      el.stageVideo.hidden = true;
      el.stageCompare.hidden = false;
      el.compareSource.currentTime = 0;
      el.compareResult.currentTime = 0;
      el.compareSource.play().catch(() => {});
      el.compareResult.play().catch(() => {});
    } else {
      el.stageCompare.hidden = true;
      el.stageVideo.hidden = false;
    }
  }

  /* ------------------------------------------------------------------ history */

  async function loadHistory() {
    try {
      const payload = await api('/api/jobs?limit=24');
      renderHistory(payload.jobs || []);
    } catch {
      /* silent: history is a convenience */
    }
  }

  function renderHistory(jobs) {
    el.historyStrip.innerHTML = '';
    const usable = jobs.filter((job) => job.urls?.thumb || job.status === 'error');
    if (!usable.length) {
      el.historyStrip.innerHTML = '<p class="history__empty">Nothing yet — your renders will collect here.</p>';
      return;
    }
    for (const job of usable) {
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'thumb';
      button.dataset.status = job.status;
      if (state.job?.id === job.id) button.classList.add('is-active');
      button.title = `${job.engine} · ${job.status}${job.error ? ' · ' + job.error : ''}`;

      if (job.urls?.thumb) {
        const img = document.createElement('img');
        img.src = mediaUrl(job.urls.thumb);
        img.alt = '';
        img.loading = 'lazy';
        button.append(img);
      }
      const label = document.createElement('span');
      label.textContent = job.status === 'error' ? 'failed' : `${Math.round(job.meta?.duration || 0)}s`;
      button.append(label);

      button.addEventListener('click', () => openJob(job.id));
      el.historyStrip.append(button);
    }
  }

  async function openJob(jobId) {
    try {
      const job = await api(`/api/jobs/${jobId}`);
      state.job = job;
      if (job.status === 'done') {
        setPill('ok', 'Ready');
        showResult(job);
      } else if (job.status === 'error') {
        showError(job.error || 'failed');
      } else if (['queued', 'running'].includes(job.status)) {
        startPolling(jobId);
      } else {
        showEmpty('Canceled', 'This job was stopped.');
      }
      updateJob(job);
      loadHistory();
    } catch (error) {
      toast(error.message, 'error');
    }
  }

  /* ------------------------------------------------------------------ reuse */

  async function reuse(job) {
    if (!job) return;
    const params = job.params || {};
    if (params.prompt != null) el.prompt.value = params.prompt;
    if (params.negative != null) el.negative.value = params.negative;
    if (params.steps) { el.steps.value = params.steps; state.touched.add('steps'); }
    if (params.cfg != null) { el.cfg.value = params.cfg; state.touched.add('cfg'); }
    if (params.fps) { el.fps.value = params.fps; state.touched.add('fps'); }
    if (params.seed != null) { el.seed.value = params.seed; state.touched.add('seed'); }
    if (params.width && params.height) {
      const key = `${params.width}x${params.height}`;
      const exists = $$('option', el.resolution).some((o) => o.value === key);
      if (exists) el.resolution.value = key;
    }
    if (params.keep_audio != null) el.keepAudio.checked = Boolean(params.keep_audio);
    if (params.audio_volume != null) {
      el.audioVolume.value = params.audio_volume;
      el.audioVolumeOut.textContent = `${Math.round(Number(params.audio_volume) * 100)}%`;
      paintRange(el.audioVolume);
    }

    // Try to restore the actual source files so "Reuse" can regenerate directly.
    try {
      const [motionBlob, charBlob] = await Promise.all([
        job.urls?.source ? fetch(mediaUrl(job.urls.source), { headers: headers() }).then((r) => r.blob()) : null,
        job.urls?.character ? fetch(mediaUrl(job.urls.character), { headers: headers() }).then((r) => r.blob()) : null,
      ]);
      if (motionBlob?.size) {
        setFile('motion', new File([motionBlob], 'motion.mp4', { type: motionBlob.type || 'video/mp4' }));
      }
      if (charBlob?.size) {
        setFile('character', new File([charBlob], 'character.png', { type: charBlob.type || 'image/png' }));
      }
      toast('Settings and sources restored.', 'ok');
    } catch {
      toast('Settings restored — re-add the source files to regenerate.', 'info');
    }
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }

  /* ------------------------------------------------------------------ settings */

  function openSettings() {
    el.setBackend.value = state.base;
    el.setToken.value = state.token;
    renderHealth(state.health);
    el.dialogSettings.showModal();
  }

  async function saveSettings() {
    state.base = normalizeBase(el.setBackend.value);
    state.token = el.setToken.value.trim();
    localStorage.setItem(LS.base, state.base);
    localStorage.setItem(LS.token, state.token);
    const health = await refreshHealth();
    if (health) {
      await loadEngines().catch(() => {});
      toast('Connected.', 'ok');
    }
  }

  /* ------------------------------------------------------------------ wiring */

  function cacheEls() {
    el.pill = $('#status-pill');
    el.gpuBadge = $('#gpu-badge');
    el.toasts = $('#toasts');
    el.health = $('#health-panel');
    el.engine = $('#engine');
    el.engineNote = $('#engine-note');
    el.seconds = $('#seconds');
    el.secondsOut = $('#seconds-out');
    el.resolution = $('#resolution');
    el.steps = $('#steps');
    el.cfg = $('#cfg');
    el.fps = $('#fps');
    el.seed = $('#seed');
    el.prompt = $('#prompt');
    el.negative = $('#negative');
    el.keepAudio = $('#keep-audio');
    el.audioNote = $('#audio-note');
    el.audioVolume = $('#audio-volume');
    el.audioVolumeOut = $('#audio-volume-out');
    el.audioVolumeWrap = $('#audio-volume-wrap');
    el.dropMotion = $('#drop-motion');
    el.dropCharacter = $('#drop-character');
    el.fileMotion = $('#file-motion');
    el.fileCharacter = $('#file-character');
    el.previewMotion = $('#preview-motion');
    el.previewCharacter = $('#preview-character');
    el.metaMotion = $('#meta-motion');
    el.metaCharacter = $('#meta-character');
    el.btnGenerate = $('#btn-generate');
    el.btnCancel = $('#btn-cancel');
    el.btnDice = $('#btn-dice');
    el.btnSettings = $('#btn-settings');
    el.btnRefresh = $('#btn-refresh');
    el.btnSelftest = $('#btn-selftest');
    el.btnReuse = $('#btn-reuse');
    el.btnDownload = $('#btn-download');
    el.download = $('#btn-download');
    el.stageEmpty = $('#stage-empty');
    el.stageVideo = $('#stage-video');
    el.stageCompare = $('#stage-compare');
    el.stageProgress = $('#stage-progress');
    el.progressBar = $('#progress-bar');
    el.progressLabel = $('#progress-label');
    el.progressSub = $('#progress-sub');
    el.resultVideo = $('#result-video');
    el.compareSource = $('#compare-source');
    el.compareResult = $('#compare-result');
    el.badgeMeta = $('#badge-meta');
    el.statEngine = $('#stat-engine');
    el.statElapsed = $('#stat-elapsed');
    el.statLength = $('#stat-length');
    el.statAudio = $('#stat-audio');
    el.logs = $('#logs');
    el.logsWrap = $('#logs-wrap');
    el.historyStrip = $('#history-strip');
    el.dialogSettings = $('#dialog-settings');
    el.setBackend = $('#set-backend');
    el.setToken = $('#set-token');
    el.btnSaveSettings = $('#btn-save-settings');
  }

  function wire() {
    bindDrop('motion');
    bindDrop('character');

    el.engine.addEventListener('change', () => {
      applyEngine(el.engine.value);
      saveForm();
    });

    for (const [node, key] of [
      [el.seconds, 'seconds'],
      [el.steps, 'steps'],
      [el.cfg, 'cfg'],
      [el.fps, 'fps'],
      [el.resolution, 'resolution'],
      [el.seed, 'seed'],
    ]) {
      node.addEventListener('input', () => {
        state.touched.add(key);
        if (key === 'fps' || key === 'seconds') updateSeconds();
        saveForm();
      });
      if (node.tagName === 'SELECT') {
        node.addEventListener('change', () => {
          state.touched.add(key);
          saveForm();
        });
      }
    }

    el.prompt.addEventListener('input', saveForm);
    el.negative.addEventListener('input', saveForm);

    el.audioVolume.addEventListener('input', () => {
      el.audioVolumeOut.textContent = `${Math.round(Number(el.audioVolume.value) * 100)}%`;
      paintRange(el.audioVolume);
    });

    el.keepAudio.addEventListener('change', () => {
      const on = el.keepAudio.checked;
      el.audioVolumeWrap.style.opacity = on ? '1' : '0.4';
      el.audioVolume.disabled = !on;
    });

    el.btnGenerate.addEventListener('click', submit);
    el.btnCancel.addEventListener('click', async () => {
      if (!state.job) return;
      try {
        await api(`/api/jobs/${state.job.id}/cancel`, { method: 'POST' });
        toast('Cancel requested.', 'info');
      } catch (error) {
        toast(error.message, 'error');
      }
    });
    el.btnDice.addEventListener('click', () => {
      el.seed.value = String(Math.floor(Math.random() * 2 ** 31));
      state.touched.add('seed');
      saveForm();
    });
    el.btnRefresh.addEventListener('click', loadHistory);
    el.btnReuse.addEventListener('click', () => reuse(state.job));

    el.btnSelftest.addEventListener('click', async () => {
      el.btnSelftest.disabled = true;
      try {
        const payload = await api('/api/selftest', { method: 'POST' });
        toast('Self-test queued — rendering a synthetic clip.', 'ok');
        startPolling(payload.job.id);
      } catch (error) {
        const problems = error.detail?.problems || error.detail;
        toast(`Self-test failed: ${Array.isArray(problems) ? problems[0] : problems || error.message}`, 'error', 9000);
      } finally {
        el.btnSelftest.disabled = false;
      }
    });

    $$('.tab').forEach((tab) =>
      tab.addEventListener('click', () => setView(tab.dataset.view))
    );

    el.btnSettings.addEventListener('click', openSettings);
    el.dialogSettings.addEventListener('close', () => {
      if (el.dialogSettings.returnValue === 'save') saveSettings();
    });

    document.addEventListener('keydown', (event) => {
      if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') submit();
    });

    window.addEventListener('beforeunload', stopPolling);
  }

  async function boot() {
    cacheEls();
    wire();
    restoreForm();
    updateSeconds();
    paintRange(el.audioVolume);

    if (location.protocol === 'file:') {
      toast('Opened from disk — set the backend URL in settings.', 'info', 7000);
    }

    const health = await refreshHealth();
    if (health) {
      await loadEngines().catch((error) => toast(error.message, 'error'));
      loadHistory();
      updateGenerateState();
    } else if (state.base || location.protocol !== 'file:') {
      openSettings();
    }

    // Periodic heartbeat so the status pill reflects the GPU going away.
    setInterval(() => {
      if (!isBusy()) refreshHealth({ quiet: true });
    }, 20000);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot);
  } else {
    boot();
  }
})();
