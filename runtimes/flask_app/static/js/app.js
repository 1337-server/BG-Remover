/*
 * Alpine.js controller powering the Flask background remover web UI.
 * Provides state management, persistence, and interaction handlers for the
 * upload forms, batch processor, previews, and activity log.
 */
const SETTINGS_STORAGE_KEY = 'bgr-flask-settings';

function readStoredSettings() {
  try {
    const rawValue = localStorage.getItem(SETTINGS_STORAGE_KEY);
    if (!rawValue) {
      return null;
    }
    return JSON.parse(rawValue);
  } catch (error) {
    console.warn('Unable to read stored settings', error);
    return null;
  }
}

function writeStoredSettings(settings) {
  try {
    localStorage.setItem(SETTINGS_STORAGE_KEY, JSON.stringify(settings));
  } catch (error) {
    console.warn('Unable to persist settings', error);
  }
}

function normaliseModelOptions(models) {
  if (!Array.isArray(models)) {
    return [];
  }
  return models
    .map((item) => ({
      key: item.key,
      display_name: item.display_name || item.key,
      defaults: item.defaults || {},
      constraints: item.constraints || {},
    }))
    .filter((item) => typeof item.key === 'string' && item.key.length > 0);
}

function formatTooltip(meta) {
  if (!meta) {
    return '';
  }
  const lines = [meta.description || ''];
  if (meta.min !== null && meta.min !== undefined) {
    lines.push(`Min: ${meta.min}`);
  }
  if (meta.max !== null && meta.max !== undefined) {
    lines.push(`Max: ${meta.max}`);
  }
  if (meta.default !== undefined && meta.default !== null) {
    lines.push(`Default: ${meta.default}`);
  }
  if (meta.recommended) {
    lines.push(`Recommended: ${meta.recommended}`);
  }
  return lines.filter(Boolean).join('\n');
}

function clampValue(value, meta) {
  if (!meta) {
    return value;
  }
  let result = value;
  if (meta.min !== null && meta.min !== undefined && result < meta.min) {
    result = meta.min;
  }
  if (meta.max !== null && meta.max !== undefined && result > meta.max) {
    result = meta.max;
  }
  return result;
}

function serialiseAdvancedSettings(settings) {
  return {
    feather_radius: Number(settings.feather_radius ?? 3),
    resize_mode: settings.resize_mode || 'stretch',
    alpha_matting: Boolean(settings.alpha_matting),
    foreground_threshold: Number(settings.foreground_threshold ?? 240),
    background_threshold: Number(settings.background_threshold ?? 10),
    erode_size: Number(settings.erode_size ?? 10),
    smoothing: Number(settings.smoothing ?? 0),
    mask_blur: Number(settings.mask_blur ?? 0),
    edge_refinement: Boolean(settings.edge_refinement),
    transparent: Boolean(settings.transparent),
    background_color: settings.background_color || '#ffffff',
    preserve_names: Boolean(settings.preserve_names),
    output_format: settings.output_format || 'PNG',
    max_workers: Number(settings.max_workers ?? 1),
    provider: settings.provider || 'auto',
    model_dir: settings.model_dir || '',
    remember_preferences: Boolean(settings.remember_preferences ?? true),
    output_directory: settings.output_directory || '',
  };
}

window.bgrApp = function bgrApp(rawConfig) {
  const config = rawConfig && typeof rawConfig === 'object' ? rawConfig : {};
  const badgeLabel = config.badge_label || 'CPU';
  const providers = Array.isArray(config.providers) ? config.providers : [];
  const defaultOutputDir = typeof config.default_output_dir === 'string' && config.default_output_dir
    ? config.default_output_dir
    : './output';

  return {
    badgeLabel,
    providers,
    defaultOutputDir,
    models: [],
    optionsHelp: {},
    settings: {},
    validationErrors: {},
    previewItems: [],
    history: [],
    activityLog: [],
    toast: { visible: false, message: '', type: 'info' },
    isReady: false,
    single: {
      files: [],
      outputDir: defaultOutputDir,
      backgroundMode: 'none',
      isProcessing: false,
      statusMessage: '',
      error: null,
      previewMode: 'none',
      lastFile: null,
    },
    batch: {
      file: null,
      fileName: '',
      outputDir: defaultOutputDir,
      isProcessing: false,
      logs: [],
      summary: null,
      jobId: null,
      eventSource: null,
      error: null,
      showSummary: false,
    },

    async init() {
      await this.bootstrap();
      this.loadHistory();
    },

    async bootstrap() {
      await Promise.all([this.fetchModels(), this.fetchOptionsHelp()]);
      this.restoreSettings();
      this.isReady = true;
    },

    async fetchModels() {
      try {
        const response = await fetch('/api/models');
        if (!response.ok) {
          throw new Error(`Failed to fetch models (${response.status})`);
        }
        const payload = await response.json();
        this.models = normaliseModelOptions(payload.models);
        this.defaultModel = payload.default_model;
      } catch (error) {
        console.error('Failed to fetch models', error);
        this.showToast('Unable to load model list. Using defaults.', 'error');
        this.models = [];
        this.defaultModel = 'isnet-general-use';
      }
    },

    async fetchOptionsHelp() {
      try {
        const response = await fetch('/api/help/options');
        if (!response.ok) {
          throw new Error(`Failed to fetch options (${response.status})`);
        }
        const payload = await response.json();
        this.optionsHelp = payload.options || {};
      } catch (error) {
        console.error('Failed to load option help', error);
        this.optionsHelp = {};
      }
    },

    computeDefaults() {
      const modelKey = (this.defaultModel && this.defaultModel.trim()) || (this.models[0] ? this.models[0].key : 'isnet-general-use');
      const help = this.optionsHelp;
      return {
        model_key: modelKey,
        feather_radius: help.feather_radius?.default ?? 3,
        resize_mode: help.resize_mode?.default ?? 'stretch',
        alpha_matting: help.alpha_matting?.default ?? false,
        foreground_threshold: help.foreground_threshold?.default ?? 240,
        background_threshold: help.background_threshold?.default ?? 10,
        erode_size: help.erode_size?.default ?? 10,
        smoothing: help.smoothing?.default ?? 0,
        mask_blur: help.mask_blur?.default ?? 0,
        edge_refinement: help.edge_refinement?.default ?? false,
        transparent: help.transparent?.default ?? true,
        background_color: help.background_color?.default ?? '#ffffff',
        preserve_names: help.preserve_names?.default ?? false,
        output_format: help.output_format?.default ?? 'PNG',
        max_workers: help.max_workers?.default ?? 1,
        provider: 'auto',
        model_dir: '',
        remember_preferences: true,
        output_directory: '',
      };
    },

    restoreSettings() {
      const stored = readStoredSettings();
      const defaults = this.computeDefaults();
      const merged = { ...defaults, ...(stored || {}) };
      if (!this.models.some((model) => model.key === merged.model_key)) {
        merged.model_key = defaults.model_key;
      }
      merged.feather_radius = clampValue(Number(merged.feather_radius ?? defaults.feather_radius), this.optionsHelp.feather_radius);
      merged.mask_blur = clampValue(Number(merged.mask_blur ?? defaults.mask_blur), this.optionsHelp.mask_blur);
      merged.smoothing = clampValue(Number(merged.smoothing ?? defaults.smoothing), this.optionsHelp.smoothing);
      merged.foreground_threshold = clampValue(Number(merged.foreground_threshold ?? defaults.foreground_threshold), this.optionsHelp.foreground_threshold);
      merged.background_threshold = clampValue(Number(merged.background_threshold ?? defaults.background_threshold), this.optionsHelp.background_threshold);
      merged.erode_size = clampValue(Number(merged.erode_size ?? defaults.erode_size), this.optionsHelp.erode_size);
      merged.max_workers = clampValue(Number(merged.max_workers ?? defaults.max_workers), this.optionsHelp.max_workers);
      this.settings = merged;
      this.single.outputDir = merged.output_directory || this.defaultOutputDir;
      this.batch.outputDir = merged.output_directory || this.defaultOutputDir;
    },

    persistSettings() {
      const payload = { ...this.settings, output_directory: this.settings.output_directory };
      writeStoredSettings(payload);
    },

    optionTooltip(key) {
      return formatTooltip(this.optionsHelp[key]);
    },

    selectedModel() {
      if (!this.settings?.model_key) {
        return null;
      }
      return this.models.find((model) => model.key === this.settings.model_key) || null;
    },

    selectedModelEntries() {
      const model = this.selectedModel();
      if (!model) {
        return [];
      }
      try {
        return Object.entries(model.defaults || {});
      } catch (error) {
        console.warn('Unable to enumerate model defaults', error);
        return [];
      }
    },

    setModel(key) {
      if (this.models.some((model) => model.key === key)) {
        this.settings.model_key = key;
        this.persistSettings();
      }
    },

    updateSingleFiles(event) {
      const files = Array.from(event.target.files || []);
      this.single.files = files;
      if (files.length) {
        this.single.lastFile = files[0];
      }
    },

    handleDrop(event) {
      const files = Array.from(event.dataTransfer?.files || []);
      this.single.files = files;
      if (files.length) {
        this.single.lastFile = files[0];
      }
    },

    clearSingleSelection() {
      this.single.files = [];
      this.single.lastFile = null;
    },

    resetSettings() {
      this.settings = this.computeDefaults();
      this.single.outputDir = this.settings.output_directory || this.defaultOutputDir;
      this.batch.outputDir = this.settings.output_directory || this.defaultOutputDir;
      this.persistSettings();
    },

    buildAdvancedPayload() {
      const serialised = serialiseAdvancedSettings(this.settings);
      serialised.output_directory = this.single.outputDir || '';
      return serialised;
    },

    validateSettings() {
      const errors = {};
      const help = this.optionsHelp;
      const radius = Number(this.settings.feather_radius ?? 3);
      if (help.feather_radius && (radius < help.feather_radius.min || radius > help.feather_radius.max)) {
        errors.feather_radius = `Feather radius must be between ${help.feather_radius.min} and ${help.feather_radius.max}`;
      }
      const blur = Number(this.settings.mask_blur ?? 0);
      if (help.mask_blur && (blur < help.mask_blur.min || blur > help.mask_blur.max)) {
        errors.mask_blur = `Mask blur must be between ${help.mask_blur.min} and ${help.mask_blur.max}`;
      }
      const smooth = Number(this.settings.smoothing ?? 0);
      if (help.smoothing && (smooth < help.smoothing.min || smooth > help.smoothing.max)) {
        errors.smoothing = `Smoothing must be between ${help.smoothing.min} and ${help.smoothing.max}`;
      }
      const fg = Number(this.settings.foreground_threshold ?? 240);
      if (help.foreground_threshold && (fg < help.foreground_threshold.min || fg > help.foreground_threshold.max)) {
        errors.foreground_threshold = `Foreground threshold must be between ${help.foreground_threshold.min} and ${help.foreground_threshold.max}`;
      }
      const bg = Number(this.settings.background_threshold ?? 10);
      if (help.background_threshold && (bg < help.background_threshold.min || bg > help.background_threshold.max)) {
        errors.background_threshold = `Background threshold must be between ${help.background_threshold.min} and ${help.background_threshold.max}`;
      }
      const erode = Number(this.settings.erode_size ?? 10);
      if (help.erode_size && (erode < help.erode_size.min || erode > help.erode_size.max)) {
        errors.erode_size = `Erode size must be between ${help.erode_size.min} and ${help.erode_size.max}`;
      }
      const workers = Number(this.settings.max_workers ?? 1);
      if (help.max_workers && (workers < help.max_workers.min || workers > help.max_workers.max)) {
        errors.max_workers = `Max workers must be between ${help.max_workers.min} and ${help.max_workers.max}`;
      }
      if (typeof this.settings.background_color === 'string' && !/^#?[0-9a-fA-F]{6}$/.test(this.settings.background_color)) {
        errors.background_color = 'Background colour must be a 6-digit hex value';
      }
      this.validationErrors = errors;
      return Object.keys(errors).length === 0;
    },

    showToast(message, type = 'info') {
      this.toast = { visible: true, message, type };
      setTimeout(() => {
        this.toast.visible = false;
      }, 4000);
    },

    logActivity(message, level = 'info') {
      const entry = {
        id: crypto.randomUUID(),
        message,
        level,
        class:
          level === 'error'
            ? 'flex items-center gap-2 rounded bg-rose-50 px-3 py-2 text-rose-600 dark:bg-rose-500/10 dark:text-rose-200'
            : 'flex items-center gap-2 rounded bg-slate-100 px-3 py-2 text-slate-700 dark:bg-slate-800 dark:text-slate-200',
        icon:
          level === 'error'
            ? 'error'
            : level === 'success'
              ? 'check_circle'
              : 'info',
        iconClass:
          level === 'error'
            ? 'text-rose-500'
            : level === 'success'
              ? 'text-emerald-500'
              : 'text-slate-400',
      };
      this.activityLog.unshift(entry);
      this.activityLog = this.activityLog.slice(0, 50);
    },

    async submitSingle(previewRequest = false) {
      if (!this.validateSettings()) {
        this.showToast('Fix validation errors before processing.', 'error');
        return;
      }
      const file = previewRequest ? this.single.lastFile : (this.single.files[0] || this.single.lastFile);
      if (!file) {
        this.showToast('Select an image first.', 'error');
        return;
      }

      this.single.isProcessing = true;
      this.single.error = null;
      this.single.statusMessage = previewRequest ? 'Updating preview…' : 'Processing image…';

      const advanced = this.buildAdvancedPayload();
      if (previewRequest) {
        advanced.output_directory = this.single.outputDir || '';
      }

      const formData = new FormData();
      formData.append('file', file, file.name || 'upload.png');
      formData.append('removal_model', this.settings.model_key);
      formData.append('background_mode', this.single.backgroundMode || 'none');
      formData.append('output_dir', this.single.outputDir || '');
      formData.append('preview', previewRequest ? 'true' : 'false');
      formData.append('advanced', JSON.stringify(advanced));

      try {
        const response = await fetch('/api/process/single', { method: 'POST', body: formData });
        const payload = await response.json();
        if (!response.ok || payload.status !== 'ok') {
          throw payload;
        }
        if (!previewRequest) {
          this.single.lastFile = file;
          this.logActivity(`Processed ${file.name} ✓`, 'success');
          this.addPreviewItem(payload, file.name);
          this.loadHistory();
        } else if (this.previewItems.length) {
          const updated = { ...this.previewItems[0] };
          updated.preview_url = `${payload.output_url}?t=${Date.now()}`;
          updated.download_url = payload.download_url;
          updated.output_path = payload.output_path;
          updated.timings = payload.timings;
          this.previewItems.splice(0, 1, updated);
          this.logActivity('Preview updated', 'info');
        }
      } catch (error) {
        const message = error?.message || 'Processing failed';
        this.single.error = error;
        this.logActivity(`Processing failed ✗ — ${message}`, 'error');
        this.showToast(message, 'error');
      } finally {
        this.single.isProcessing = false;
        this.single.statusMessage = '';
      }
    },

    addPreviewItem(payload, originalName) {
      const url = new URL(payload.output_url, window.location.origin);
      const identifier = url.pathname.split('/').filter(Boolean).pop() || crypto.randomUUID();
      const preview = {
        id: identifier,
        original_name: originalName,
        preview_url: payload.output_url,
        download_url: payload.download_url,
        output_path: payload.output_path,
        model_used: payload.model_used,
        timings: payload.timings,
      };
      this.previewItems.unshift(preview);
      this.previewItems = this.previewItems.slice(0, 10);
    },

    async setPreviewMode(mode) {
      if (this.single.backgroundMode === mode) {
        return;
      }
      this.single.backgroundMode = mode;
      if (this.single.lastFile) {
        await this.submitSingle(true);
      }
    },

    async submitBatch() {
      if (!this.validateSettings()) {
        this.showToast('Fix validation errors before starting batch.', 'error');
        return;
      }
      if (!this.batch.file) {
        this.showToast('Upload a ZIP archive for batch processing.', 'error');
        return;
      }
      this.batch.isProcessing = true;
      this.batch.logs = [];
      this.batch.error = null;
      this.batch.summary = null;

      const formData = new FormData();
      formData.append('file', this.batch.file, this.batch.file.name || 'batch.zip');
      formData.append('removal_model', this.settings.model_key);
      formData.append('output_dir', this.batch.outputDir || '');
      formData.append('background_mode', this.single.backgroundMode || 'none');
      formData.append('advanced', JSON.stringify(serialiseAdvancedSettings({ ...this.settings, output_directory: this.batch.outputDir || '' })));

      try {
        const response = await fetch('/api/process/batch', { method: 'POST', body: formData });
        const payload = await response.json();
        if (!response.ok || payload.status !== 'accepted') {
          throw payload;
        }
        this.batch.jobId = payload.job_id;
        this.attachEventSource(payload.job_id);
        this.logBatch('Batch job accepted.', 'info');
      } catch (error) {
        const message = error?.message || 'Batch submission failed';
        this.batch.error = error;
        this.batch.isProcessing = false;
        this.showToast(message, 'error');
        this.logBatch(`Batch failed to start — ${message}`, 'error');
      }
    },

    attachEventSource(jobId) {
      if (this.batch.eventSource) {
        this.batch.eventSource.close();
      }
      const source = new EventSource(`/api/stream/batch/${jobId}`);
      this.batch.eventSource = source;

      source.addEventListener('started', (event) => {
        try {
          const data = JSON.parse(event.data || '{}');
          this.logBatch(`Batch started — ${data.total ?? 0} items`, 'info');
        } catch (error) {
          console.warn('Failed to parse started event', error);
        }
      });

      source.addEventListener('item_success', (event) => {
        try {
          const data = JSON.parse(event.data || '{}');
          this.logBatch(`${data.filename} processed successfully ✓`, 'success');
        } catch (error) {
          console.warn('Failed to parse success event', error);
        }
      });

      source.addEventListener('item_error', (event) => {
        try {
          const data = JSON.parse(event.data || '{}');
          this.logBatch(`${data.filename} failed ✗ — Reason: ${data.error}`, 'error');
        } catch (error) {
          console.warn('Failed to parse error event', error);
        }
      });

      source.addEventListener('finished', async (event) => {
        try {
          const data = JSON.parse(event.data || '{}');
          if (data.status === 'failed') {
            this.logBatch(`Batch failed — ${data.error || 'Unknown error'}`, 'error');
            this.showToast(data.error || 'Batch failed', 'error');
          } else {
            this.logBatch('Batch finished ✓', 'success');
          }
        } catch (error) {
          console.warn('Failed to parse finished event', error);
        }
        source.close();
        this.batch.eventSource = null;
        this.batch.isProcessing = false;
        if (this.batch.jobId) {
          this.fetchBatchSummary(this.batch.jobId);
        }
      });

      source.onerror = (event) => {
        console.warn('SSE connection error', event);
        this.logBatch('Lost connection to batch stream.', 'error');
      };
    },

    async fetchBatchSummary(jobId) {
      try {
        const response = await fetch(`/api/jobs/${jobId}/summary`);
        if (!response.ok) {
          throw new Error(`Summary request failed (${response.status})`);
        }
        const payload = await response.json();
        if (payload.status === 'ok') {
          this.batch.summary = payload.job;
          this.batch.showSummary = true;
        }
      } catch (error) {
        console.warn('Failed to fetch batch summary', error);
      }
    },

    resetBatch() {
      if (this.batch.eventSource) {
        this.batch.eventSource.close();
        this.batch.eventSource = null;
      }
      this.batch.file = null;
      this.batch.fileName = '';
      this.batch.logs = [];
      this.batch.summary = null;
      this.batch.error = null;
      this.batch.isProcessing = false;
      this.batch.jobId = null;
      this.batch.showSummary = false;
    },

    logBatch(message, level = 'info') {
      const entry = {
        id: crypto.randomUUID(),
        message,
        level,
        class:
          level === 'error'
            ? 'border-l-4 border-rose-500 bg-rose-50 px-3 py-2 text-sm text-rose-600 dark:bg-rose-500/10 dark:text-rose-200'
            : level === 'success'
              ? 'border-l-4 border-emerald-500 bg-emerald-50 px-3 py-2 text-sm text-emerald-600 dark:bg-emerald-500/10 dark:text-emerald-200'
              : 'border-l-4 border-slate-400 bg-slate-100 px-3 py-2 text-sm text-slate-700 dark:bg-slate-800 dark:text-slate-200',
      };
      this.batch.logs.push(entry);
      this.batch.logs = this.batch.logs.slice(-200);
    },

    updateBatchFile(event) {
      const files = Array.from(event.target.files || []);
      this.batch.file = files[0] || null;
      this.batch.fileName = this.batch.file ? this.batch.file.name : '';
    },

    async loadHistory() {
      try {
        const response = await fetch('/history');
        if (!response.ok) {
          throw new Error(`History request failed (${response.status})`);
        }
        const payload = await response.json();
        this.history = Array.isArray(payload.history) ? payload.history : [];
      } catch (error) {
        console.warn('Failed to load history', error);
      }
    },

    openInNewTab(identifier) {
      window.open(`/result/${identifier}`, '_blank');
    },

    removePreview(identifier) {
      this.previewItems = this.previewItems.filter((item) => item.id !== identifier);
    },

    copyErrorDetails(error) {
      try {
        navigator.clipboard.writeText(JSON.stringify(error, null, 2));
        this.showToast('Error details copied to clipboard.', 'success');
      } catch (clipboardError) {
        console.warn('Unable to copy error details', clipboardError);
      }
    },

    formatBytes(bytes) {
      if (!Number.isFinite(bytes)) {
        return '-';
      }
      if (bytes === 0) {
        return '0 B';
      }
      const k = 1024;
      const sizes = ['B', 'KB', 'MB', 'GB'];
      const i = Math.min(Math.floor(Math.log(bytes) / Math.log(k)), sizes.length - 1);
      const value = bytes / k ** i;
      return `${value.toFixed(1)} ${sizes[i]}`;
    },

    formatDate(value) {
      if (!value) {
        return '-';
      }
      const date = new Date(value);
      if (Number.isNaN(date.getTime())) {
        return value;
      }
      return date.toLocaleString();
    },
  };
};
