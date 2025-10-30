/*
 * Alpine.js controller powering the Flask background remover web UI.
 * Provides state management, persistence, and interaction handlers for the
 * upload forms, batch processor, previews, and activity log.
 */
const SETTINGS_STORAGE_KEY = 'bgr-flask-settings';

/**
 * Resolve the current theme label with graceful fallback when the theme helper
 * script is unavailable.
 *
 * @returns {string} Human readable theme label.
 */
function resolveThemeLabel() {
  if (typeof window !== 'undefined' && typeof window.currentTheme === 'function') {
    return window.currentTheme() === 'dark' ? 'Dark' : 'Light';
  }
  return document.documentElement.classList.contains('dark') ? 'Dark' : 'Light';
}

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

/**
 * Create the Alpine.js application state tree.
 *
 * Exposing the factory on `window` ensures Alpine can invoke it directly from
 * the HTML `x-data` attribute regardless of script execution order.
 *
 * @param {string|object} rawConfig - JSON string or object containing template provided values.
 * @returns {object} Alpine component descriptor consumed by the UI templates.
 */
window.bgrApp = function bgrApp(rawConfig) {
  console.log('✅ bgrApp registered globally:', typeof window.bgrApp);

  let parsedConfig = {};
  try {
    if (typeof rawConfig === 'string') {
      parsedConfig = JSON.parse(rawConfig);
    } else if (rawConfig && typeof rawConfig === 'object') {
      parsedConfig = { ...rawConfig };
    }
  } catch (error) {
    console.error('Config parse failed:', error, rawConfig);
    parsedConfig = {};
  }

  const normalizedProviders = Array.isArray(parsedConfig.providers) ? parsedConfig.providers : [];
  const normalizedModels = Array.isArray(parsedConfig.model_options) ? parsedConfig.model_options : [];
  const normalizedModelDir = typeof parsedConfig.model_dir === 'string' ? parsedConfig.model_dir : '';
  const normalizedDefaultModel =
    typeof parsedConfig.default_model === 'string' && parsedConfig.default_model
      ? parsedConfig.default_model
      : 'isnet-general-use';
  const normalizedModelSpecs =
    parsedConfig.model_specs && typeof parsedConfig.model_specs === 'object'
      ? { ...parsedConfig.model_specs }
      : {};
  const normalizedBadgeLabel = parsedConfig.badge_label || normalizedProviders[0] || 'CPU';

  const baseDefaults = {
    model_key: normalizedDefaultModel,
    removal_model: normalizedDefaultModel,
    model_precision: 'auto',
    feather_radius: 3,
    provider: 'auto',
    background_color: '#ffffff',
    background_mode: 'clear',
    transparent: true,
    output_format: 'PNG',
    model_dir: normalizedModelDir,
    output_dir: '',
    preserve_names: false,
    alpha_matting: false,
    alpha_matting_foreground_threshold: 240,
    alpha_matting_background_threshold: 10,
    alpha_matting_erode_size: 10,
    post_process_mask: true,
    only_mask: false,
    mask_blur: 0,
    mask_threshold: 0,
    cut_out_mode: 'object',
    remember_preferences: true,
  };

  const createDefaults = () => ({
    ...baseDefaults,
    model_key: normalizedModels.length ? normalizedModels[0] : baseDefaults.model_key,
    removal_model: normalizedModels.length ? normalizedModels[0] : baseDefaults.model_key,
    model_dir: normalizedModelDir,
  });

  const createEmptyBatchSummary = () => ({
    total: 0,
    success: 0,
    failed: 0,
    size_bytes: 0,
    download_url: '',
    output_dir: '',
    status: 'idle',
    message: '',
  });

  const SUPPORTED_BATCH_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.webp'];

  const buildInitialSettings = () => {
    const stored = readStoredSettings();
    const defaults = createDefaults();
    if (!stored) {
      return defaults;
    }
    if (stored.output_directory && !stored.output_dir) {
      stored.output_dir = stored.output_directory;
    }
    return {
      ...defaults,
      ...stored,
      model_key: stored.model_key || defaults.model_key,
      removal_model: stored.removal_model || stored.model_key || defaults.model_key,
      model_dir: stored.model_dir ?? defaults.model_dir,
      provider: stored.provider || defaults.provider,
    };
  };

  return {
    initialConfig: parsedConfig,
    badgeLabel: normalizedBadgeLabel,
    providers: normalizedProviders,
    provider: normalizedBadgeLabel,
    modelOptions: normalizedModels,
    modelSpecs: normalizedModelSpecs,
    modelDir: normalizedModelDir,
    providerPill: normalizedBadgeLabel,
    providerPillClass: '',
    themeLabel: resolveThemeLabel(),
    selectedFiles: [],
    previewItems: [],
    history: [],
    activityLog: [],
    batchLog: [],
    isProcessing: false,
    isBatchProcessing: false,
    isDragging: false,
    batchFile: null,
    batchFileName: '',
    batchMode: 'zip',
    batchFolderFiles: [],
    batchFolderLabel: '',
    batchServerFolderPath: '',
    batchServerFolderLabel: '',
    batchImageCount: 0,
    batchRecursive: false,
    batchJobId: '',
    batchEventSource: null,
    batchOutputFiles: [],
    batchShowFiles: false,
    batchSummary: createEmptyBatchSummary(),
    settings: buildInitialSettings(),
    optionHelp: {},
    status: { message: 'Ready', tone: 'ready' },
    previewBackground: 'clear',

    init() {
      this.providerPillClass = this.computeProviderClass(this.providerPill);
      this.applyTheme();
      this.ensureModelDefaults();
      this.normaliseBackgroundSettings();
      this.previewBackground = this.settings.background_mode === 'fill' ? 'fill' : 'clear';
      this.persistSettings();
      this.loadHistory();
      this.fetchOptionHelp();
      this.status = { message: 'Ready', tone: 'ready' };
      this.$watch('settings.model_key', (value) => {
        if (value) {
          this.settings.removal_model = value;
          this.persistSettings();
        }
      });
      this.$watch('batchMode', (mode) => {
        switch (mode) {
          case 'zip':
            this.batchFolderFiles = [];
            this.batchFolderLabel = '';
            this.batchServerFolderPath = '';
            this.batchServerFolderLabel = '';
            this.batchImageCount = 0;
            {
              const folderInput = document.getElementById('batch-folder-input');
              if (folderInput) {
                folderInput.value = '';
              }
            }
            break;
          case 'folder':
            this.batchFile = null;
            this.batchFileName = '';
            this.batchServerFolderPath = '';
            this.batchServerFolderLabel = '';
            {
              const zipInput = document.getElementById('batch-input');
              if (zipInput) {
                zipInput.value = '';
              }
            }
            break;
          case 'server':
            this.batchFile = null;
            this.batchFileName = '';
            this.batchFolderFiles = [];
            this.batchFolderLabel = '';
            this.batchImageCount = 0;
            {
              const folderInput = document.getElementById('batch-folder-input');
              if (folderInput) {
                folderInput.value = '';
              }
            }
            {
              const zipInput = document.getElementById('batch-input');
              if (zipInput) {
                zipInput.value = '';
              }
            }
            break;
          default:
            break;
        }
      });
      console.log('✅ Alpine initialized successfully with config:', this.initialConfig);
    },

    async fetchOptionHelp() {
      try {
        const response = await fetch('/api/help/options');
        if (!response.ok) {
          throw new Error('Unable to load option help');
        }
        const payload = await response.json();
        if (payload && typeof payload === 'object' && payload.options) {
          this.optionHelp = payload.options;
        }
      } catch (error) {
        console.warn('Option help request failed', error);
      }
    },

    getOptionHelp(key) {
      if (!key) {
        return '';
      }
      return this.optionHelp?.[key] || '';
    },

    activeModelSpecs() {
      if (!this.settings || !this.settings.model_key) {
        return null;
      }
      return this.modelSpecs?.[this.settings.model_key] || null;
    },

    activeModelSpecsEntries() {
      const specs = this.activeModelSpecs();
      if (!specs) {
        return [];
      }
      try {
        return Object.entries(specs);
      } catch (error) {
        console.warn('Unable to enumerate model specs', error);
        return [];
      }
    },

    ensureModelDefaults() {
      if (
        (!this.settings.model_key || !this.modelOptions.includes(this.settings.model_key)) &&
        this.modelOptions.length
      ) {
        this.settings.model_key = this.modelOptions[0];
      }
      if (!this.settings.model_dir) {
        this.settings.model_dir = this.modelDir;
      }
      if (!this.settings.provider || !['auto', 'gpu', 'cpu'].includes(this.settings.provider)) {
        this.settings.provider = (this.provider || 'auto').toLowerCase();
      }
      if (!this.settings.removal_model) {
        this.settings.removal_model = this.settings.model_key;
      }
      if (!this.settings.background_mode) {
        this.settings.background_mode = 'clear';
      }
    },

    normaliseBackgroundSettings() {
      if (!this.settings) {
        return;
      }
      if (this.settings.background_mode === 'fill') {
        this.settings.transparent = false;
      } else if (this.settings.background_mode === 'clear') {
        this.settings.transparent = true;
      }
      if (!['object', 'mask', 'bbox'].includes(this.settings.cut_out_mode)) {
        this.settings.cut_out_mode = 'object';
      }
    },

    computeProviderClass(label) {
      const base = 'provider-pill';
      if (label === 'GPU') {
        return `${base} provider-pill--gpu`;
      }
      return `${base} provider-pill--default`;
    },

    applyTheme() {
      this.themeLabel = resolveThemeLabel();
    },

    toggleTheme() {
      let theme = 'light';
      if (typeof window !== 'undefined' && typeof window.toggleTheme === 'function') {
        theme = window.toggleTheme();
      } else {
        const element = document.documentElement;
        const isDark = element.classList.toggle('dark');
        theme = isDark ? 'dark' : 'light';
        localStorage.setItem('bgr-theme', theme);
        element.dataset.theme = theme;
        element.style.colorScheme = theme;
      }
      this.themeLabel = theme === 'dark' ? 'Dark' : 'Light';
    },

    setStatus(message, tone = 'ready') {
      this.status = { message, tone };
    },

    statusClass() {
      const tone = this.status?.tone || 'ready';
      return {
        'status-pill': true,
        'status-pill--processing': tone === 'processing',
        'status-pill--error': tone === 'error',
        'status-pill--ready': tone === 'ready',
      };
    },

    previewBackgroundClasses() {
      return {
        'preview-stage': true,
        'preview-transparent': this.previewBackground !== 'fill',
      };
    },

    previewContainerStyle() {
      if (this.previewBackground === 'fill') {
        const colour = this.settings.background_color || '#ffffff';
        return { backgroundColor: colour };
      }
      return {};
    },

    applyPreviewFill() {
      this.previewBackground = 'fill';
    },

    applyPreviewClear() {
      this.previewBackground = 'clear';
    },

    createLogEntry(level, message) {
      const id = `${Date.now()}-${Math.random().toString(16).slice(2)}`;
      const iconMap = {
        success: { icon: 'task_alt', iconClass: 'log-icon--success' },
        error: { icon: 'error', iconClass: 'log-icon--error' },
        info: { icon: 'info', iconClass: 'log-icon--info' },
      };
      const meta = iconMap[level] || iconMap.info;
      return { id, message, icon: meta.icon, iconClass: meta.iconClass, class: level };
    },

    persistSettings() {
      try {
        this.settings.removal_model = this.settings.model_key;
        this.normaliseBackgroundSettings();
        localStorage.setItem(SETTINGS_STORAGE_KEY, JSON.stringify(this.settings));
      } catch (error) {
        console.warn('Unable to persist settings', error);
      }
    },

    resetSettings() {
      this.settings = createDefaults();
      this.normaliseBackgroundSettings();
      this.persistSettings();
      this.previewBackground = this.settings.background_mode === 'fill' ? 'fill' : 'clear';
    },

    updateSelections(event) {
      const files = Array.from(event?.target?.files || []);
      this.selectedFiles = files;
    },

    handleDrop(event) {
      this.isDragging = false;
      const files = Array.from(event?.dataTransfer?.files || []).filter((file) => file.type.startsWith('image/'));
      if (!files.length) {
        this.showToast('info', 'Drop supported image files.');
        return;
      }
      this.selectedFiles = files;
    },

    resetSelections() {
      this.selectedFiles = [];
      const input = document.getElementById('image-input');
      if (input) {
        input.value = '';
      }
    },

    updateBatchSelection(event) {
      const [file] = Array.from(event?.target?.files || []);
      this.batchMode = 'zip';
      this.batchFile = file || null;
      this.batchFileName = file ? file.name : '';
      if (file) {
        this.batchFolderFiles = [];
        this.batchFolderLabel = '';
        this.batchServerFolderPath = '';
        this.batchServerFolderLabel = '';
        this.batchImageCount = 0;
      }
    },

    updateBatchFolderSelection(event) {
      const files = Array.from(event?.target?.files || []);
      this.batchMode = 'folder';
      this.batchFolderFiles = files;
      this.batchFile = null;
      this.batchFileName = '';
      this.batchServerFolderPath = '';
      this.batchServerFolderLabel = '';
      this.batchSummary = createEmptyBatchSummary();
      this.batchOutputFiles = [];
      this.batchShowFiles = false;
      if (!files.length) {
        this.batchFolderLabel = '';
        this.batchImageCount = 0;
        return;
      }
      this.batchImageCount = files.filter((file) => this.isSupportedBatchFile(file.name)).length;
      const relative = files[0]?.webkitRelativePath || files[0]?.name || '';
      const parts = relative.split(/[\\/]/).filter(Boolean);
      this.batchFolderLabel = parts.length > 1 ? parts[0] : parts[0] || 'Selected files';
      if (!this.batchImageCount) {
        this.addBatchLog('info', 'No supported images found in the selected folder.');
      } else {
        this.addBatchLog('info', `Prepared ${this.batchImageCount} image(s) from ${this.batchFolderLabel}.`);
      }
    },

    updateServerFolderPath(value) {
      const trimmed = (value || '').trim();
      this.batchServerFolderPath = trimmed;
      this.batchSummary = createEmptyBatchSummary();
      this.batchOutputFiles = [];
      this.batchShowFiles = false;
      this.batchFolderFiles = [];
      this.batchFolderLabel = '';
      this.batchImageCount = 0;
      if (!trimmed) {
        this.batchServerFolderLabel = '';
        return;
      }
      const segments = trimmed.replace(/\\/g, '/').split('/').filter(Boolean);
      this.batchServerFolderLabel = segments.length ? segments[segments.length - 1] : trimmed;
      this.addBatchLog('info', `Server folder selected: ${trimmed}`);
    },

    resetBatch() {
      this.closeBatchStream();
      this.batchFile = null;
      this.batchFileName = '';
      this.batchFolderFiles = [];
      this.batchFolderLabel = '';
      this.batchServerFolderPath = '';
      this.batchServerFolderLabel = '';
      this.batchImageCount = 0;
      this.batchRecursive = false;
      this.batchJobId = '';
      this.batchSummary = createEmptyBatchSummary();
      this.batchOutputFiles = [];
      this.batchShowFiles = false;
      this.batchLog = [];
      const input = document.getElementById('batch-input');
      if (input) {
        input.value = '';
      }
      const folderInput = document.getElementById('batch-folder-input');
      if (folderInput) {
        folderInput.value = '';
      }
      const serverInput = document.getElementById('batch-server-input');
      if (serverInput) {
        serverInput.value = '';
      }
    },

    closeBatchStream() {
      if (this.batchEventSource) {
        this.batchEventSource.close();
        this.batchEventSource = null;
      }
    },

    async processImages() {
      if (!this.selectedFiles.length) {
        this.showToast('error', 'Choose at least one image to process.');
        return;
      }
      const form = new FormData();
      this.selectedFiles.forEach((file) => form.append('images', file, file.name));
      Object.entries(this.settings).forEach(([key, value]) => {
        if (typeof value === 'boolean') {
          form.append(key, value ? 'true' : 'false');
        } else {
          form.append(key, value ?? '');
        }
      });

      this.isProcessing = true;
      this.setStatus('Processing…', 'processing');
      this.addActivity('info', `Starting processing for ${this.selectedFiles.length} image(s)…`);
      try {
        const response = await fetch('/process', {
          method: 'POST',
          body: form,
        });
        const payload = await response.json();
        if (!response.ok) {
          const errorMessage = payload.message || payload.error || 'Processing failed';
          throw new Error(errorMessage);
        }
        const results = Array.isArray(payload.results) ? payload.results : [];
        results.forEach((item) => {
          this.addActivity('success', `${item.result_name} processed successfully ✓`);
        });
        this.previewItems = [...results, ...this.previewItems].slice(0, 10);
        this.showToast('success', `Processed ${results.length} image(s) successfully.`);
        this.loadHistory();
        this.setStatus('Ready', 'ready');
      } catch (error) {
        console.error(error);
        this.addActivity('error', `Processing failed ✗ — Reason: ${error.message}`);
        this.showToast('error', error.message);
        this.setStatus(`Error: ${error.message}`, 'error');
      } finally {
        this.isProcessing = false;
      }
    },

    async processBatch() {
      const usingFolderUpload = this.batchMode === 'folder';
      const usingServerFolder = this.batchMode === 'server';
      if (usingFolderUpload && !this.batchFolderFiles.length) {
        this.showToast('error', 'Select a folder containing images to process.');
        return;
      }
      if (usingServerFolder && !this.batchServerFolderPath) {
        this.showToast('error', 'Enter a valid server folder path to process.');
        return;
      }
      if (!usingFolderUpload && !usingServerFolder && !this.batchFile) {
        this.showToast('error', 'Select a ZIP archive to process.');
        return;
      }

      const form = new FormData();
      if (usingFolderUpload) {
        this.batchFolderFiles.forEach((file) => {
          const relativePath = file.webkitRelativePath || file.name;
          form.append('folder_files', file, relativePath);
        });
        if (this.batchFolderLabel) {
          form.append('folder_path', this.batchFolderLabel);
        }
      } else if (usingServerFolder) {
        form.append('folder_path', this.batchServerFolderPath);
      } else if (this.batchFile) {
        form.append('zip_file', this.batchFile, this.batchFile.name);
      }
      form.append('recursive', this.batchRecursive ? 'true' : 'false');
      Object.entries(this.settings).forEach(([key, value]) => {
        if (typeof value === 'boolean') {
          form.append(key, value ? 'true' : 'false');
        } else {
          form.append(key, value ?? '');
        }
      });

      this.closeBatchStream();
      this.isBatchProcessing = true;
      this.batchLog = [];
      this.batchOutputFiles = [];
      this.batchShowFiles = false;
      this.batchSummary = createEmptyBatchSummary();
      const descriptor = usingFolderUpload
        ? this.batchFolderLabel || 'selected folder'
        : usingServerFolder
        ? this.batchServerFolderLabel || this.batchServerFolderPath || 'server folder'
        : this.batchFileName || 'selected archive';
      this.setStatus('Processing…', 'processing');
      this.addActivity('info', `Batch processing started for ${descriptor}…`);
      this.addBatchLog('info', `Batch processing started for ${descriptor}`);
      try {
        const response = await fetch('/api/process/batch', { method: 'POST', body: form });
        const payload = await response.json();
        if (!response.ok) {
          throw new Error(payload.error || 'Batch processing failed');
        }
        this.batchJobId = payload.job_id;
        this.batchSummary.status = 'processing';
        this.batchSummary.message = 'Processing images…';
        if (typeof payload.total === 'number') {
          this.batchSummary.total = payload.total;
        } else if (usingFolderUpload) {
          this.batchSummary.total = this.batchImageCount;
        }
        this.listenToBatchJob(this.batchJobId);
      } catch (error) {
        console.error(error);
        this.addActivity('error', `Batch failed ✗ — Reason: ${error.message}`);
        this.showToast('error', error.message);
        this.addBatchLog('error', `Batch failed — ${error.message}`);
        this.setStatus(`Error: ${error.message}`, 'error');
        this.isBatchProcessing = false;
      }
    },

    listenToBatchJob(jobId) {
      if (!jobId) {
        return;
      }
      this.closeBatchStream();
      const source = new EventSource(`/api/process/batch/${jobId}/stream`);
      this.batchEventSource = source;
      source.addEventListener('started', (event) => {
        try {
          const data = JSON.parse(event.data || '{}');
          if (typeof data.total === 'number') {
            this.batchSummary.total = data.total;
          }
          const label = data.label ? `“${data.label}”` : 'batch';
          this.addBatchLog('info', `Batch job started for ${label}.`);
        } catch (error) {
          console.warn('Unable to parse batch start payload', error);
        }
      });
      source.addEventListener('item_success', (event) => {
        try {
          const data = JSON.parse(event.data || '{}');
          const message = data.output
            ? `${data.input || 'Image'} processed → ${data.output}`
            : `${data.input || 'Image'} processed successfully.`;
          this.addBatchLog('success', `${message} ✓`);
          this.batchSummary.success += 1;
        } catch (error) {
          console.warn('Unable to parse success event', error);
        }
      });
      source.addEventListener('item_error', (event) => {
        try {
          const data = JSON.parse(event.data || '{}');
          const reason = data.error || 'Unknown error';
          const label = data.input || 'Image';
          this.addBatchLog('error', `${label} failed ✗ — Reason: ${reason}`);
          this.batchSummary.failed += 1;
        } catch (error) {
          console.warn('Unable to parse error event', error);
        }
      });
      source.addEventListener('finished', (event) => {
        try {
          const data = JSON.parse(event.data || '{}');
          this.handleBatchFinished(data);
        } catch (error) {
          console.warn('Unable to parse finished event', error);
          this.handleBatchFinished({ status: 'error', message: 'Batch finished with an invalid response.' });
        }
      });
      source.onerror = (event) => {
        console.warn('Batch stream error', event);
        if (this.isBatchProcessing) {
          this.addBatchLog('error', 'Connection to batch progress stream lost.');
          this.setStatus('Error: connection lost', 'error');
          this.isBatchProcessing = false;
        }
        this.closeBatchStream();
      };
    },

    handleBatchFinished(data) {
      this.isBatchProcessing = false;
      this.closeBatchStream();
      const summary = data.summary || {};
      this.batchSummary = {
        ...createEmptyBatchSummary(),
        ...summary,
        download_url: data.download_url || summary.download_url || '',
        output_dir: data.output_dir || summary.output_dir || this.batchSummary.output_dir,
        status: data.status || 'finished',
        message: data.message || '',
      };
      switch (this.batchSummary.status) {
        case 'error':
          this.addBatchLog('error', this.batchSummary.message || 'Batch failed.');
          this.showToast('error', this.batchSummary.message || 'Batch failed.');
          this.setStatus(`Error: ${this.batchSummary.message || 'Batch failed.'}`, 'error');
          break;
        case 'cancelled':
          if (!this.batchSummary.message) {
            this.batchSummary.message = 'Batch cancelled by client.';
          }
          this.addBatchLog('info', 'Batch cancelled by client.');
          this.showToast('info', 'Batch cancelled.');
          this.setStatus('Batch cancelled', 'ready');
          break;
        default:
          if (!this.batchSummary.message) {
            this.batchSummary.message = this.batchSummary.failed
              ? 'Batch completed with some errors.'
              : 'Batch completed successfully.';
          }
          this.addBatchLog(
            this.batchSummary.failed ? 'info' : 'success',
            `Batch completed — ${this.batchSummary.success} succeeded, ${this.batchSummary.failed} failed.`,
          );
          this.showToast(
            'success',
            `Batch completed — ${this.batchSummary.success} succeeded, ${this.batchSummary.failed} failed.`,
          );
          this.setStatus('Ready', 'ready');
      }
      this.loadHistory();
    },

    async fetchBatchFiles() {
      if (!this.batchJobId) {
        return;
      }
      try {
        const response = await fetch(`/api/process/batch/${this.batchJobId}/files`);
        const payload = await response.json();
        if (!response.ok) {
          throw new Error(payload.error || 'Unable to open output folder');
        }
        this.batchOutputFiles = Array.isArray(payload.files) ? payload.files : [];
        this.batchSummary.output_dir = payload.output_dir || this.batchSummary.output_dir;
        this.batchShowFiles = true;
      } catch (error) {
        console.error(error);
        this.showToast('error', error.message);
        this.addBatchLog('error', `Unable to open output folder — ${error.message}`);
      }
    },

    batchSelectionSummary() {
      if (this.batchMode === 'folder') {
        if (!this.batchFolderFiles.length) {
          return 'No folder selected yet.';
        }
        const label = this.batchFolderLabel || 'selected folder';
        return `${this.batchImageCount} supported image(s) in ${label}.`;
      }
      if (this.batchMode === 'server') {
        if (!this.batchServerFolderPath) {
          return 'No server folder specified yet.';
        }
        return `Server folder: ${this.batchServerFolderPath}`;
      }
      return this.batchFileName || 'No archive selected yet.';
    },

    isSupportedBatchFile(name) {
      if (!name) {
        return false;
      }
      const lowered = name.toLowerCase();
      return SUPPORTED_BATCH_EXTENSIONS.some((ext) => lowered.endsWith(ext));
    },

    removePreview(identifier) {
      this.previewItems = this.previewItems.filter((item) => item.id !== identifier);
    },

    openInNewTab(identifier) {
      window.open(`/result/${identifier}`, '_blank');
    },

    async loadHistory() {
      try {
        const response = await fetch('/history');
        if (!response.ok) {
          throw new Error('Unable to load history');
        }
        const payload = await response.json();
        this.history = payload.history || [];
      } catch (error) {
        console.error(error);
        this.showToast('error', error.message);
      }
    },

    addActivity(level, message) {
      const entry = this.createLogEntry(level, message);
      this.activityLog = [entry, ...this.activityLog].slice(0, 50);
    },

    addBatchLog(level, message) {
      const entry = this.createLogEntry(level, message);
      this.batchLog = [entry, ...this.batchLog].slice(0, 100);
    },

    showToast(type, message) {
      const container = document.getElementById('toast-container');
      if (!container) {
        return;
      }
      const toast = document.createElement('div');
      toast.className = `toast ${type}`;
      const icon = type === 'success' ? 'check_circle' : type === 'error' ? 'error' : 'info';
      toast.innerHTML = `
        <span class="material-symbols-rounded text-base">${icon}</span>
        <span>${message}</span>
      `;
      container.appendChild(toast);
      setTimeout(() => {
        toast.classList.add('opacity-0', 'transition', 'duration-300');
        setTimeout(() => toast.remove(), 320);
      }, 5000);
    },

    formatBytes(bytes) {
      if (!bytes && bytes !== 0) return '—';
      if (bytes < 1024) return `${bytes} B`;
      const units = ['KB', 'MB', 'GB'];
      let value = bytes;
      let index = -1;
      do {
        value /= 1024;
        index += 1;
      } while (value >= 1024 && index < units.length - 1);
      return `${value.toFixed(1)} ${units[index]}`;
    },

    formatDate(isoString) {
      if (!isoString) return '—';
      try {
        const date = new Date(isoString);
        return date.toLocaleString();
      } catch (error) {
        return isoString;
      }
    },
  };
};

