function appController(badgeLabel, providerList, modelOptions, defaultModelDir) {
  const defaults = {
    model_key: modelOptions?.[0] || 'isnet-general-use',
    feather_radius: 3,
    provider: 'auto',
    background_color: '#ffffff',
    transparent: true,
    output_format: 'PNG',
    model_dir: defaultModelDir || '',
    output_directory: '',
    preserve_names: false,
    alpha_matting: false,
    mask_blur: 0,
    remember_preferences: true,
  };

  let storedSettings;
  try {
    storedSettings = JSON.parse(localStorage.getItem('bgr-flask-settings') || 'null');
  } catch (error) {
    storedSettings = null;
  }

  return {
    badgeLabel: badgeLabel || 'CPU',
    providers: providerList || [],
    modelOptions: modelOptions || [],
    providerPill: badgeLabel || 'CPU',
    providerPillClass: '',
    themeLabel: 'Light',
    selectedFiles: [],
    previewItems: [],
    history: [],
    activityLog: [],
    isProcessing: false,
    isBatchProcessing: false,
    isDragging: false,
    batchFile: null,
    batchFileName: '',
    batchSummary: null,
    settings: Object.assign({}, defaults, storedSettings || {}),

    init() {
      this.providerPillClass = this.computeProviderClass(this.providerPill);
      this.themeLabel = document.documentElement.classList.contains('dark') ? 'Dark' : 'Light';
      if (!this.settings.model_key && this.modelOptions.length) {
        this.settings.model_key = this.modelOptions[0];
      }
      if (!this.settings.model_dir) {
        this.settings.model_dir = defaultModelDir || '';
      }
      this.persistSettings();
      this.loadHistory();
    },

    computeProviderClass(label) {
      const base = 'px-3 py-1 rounded-full text-xs font-semibold';
      if (label === 'GPU') {
        return `${base} bg-emerald-200/70 text-emerald-700 dark:bg-emerald-500/20 dark:text-emerald-200`;
      }
      return `${base} bg-slate-200/70 text-slate-600 dark:bg-slate-700/40 dark:text-slate-200`;
    },

    toggleTheme() {
      const element = document.documentElement;
      const isDark = element.classList.toggle('dark');
      localStorage.setItem('bgr-theme', isDark ? 'dark' : 'light');
      this.themeLabel = isDark ? 'Dark' : 'Light';
    },

    persistSettings() {
      localStorage.setItem('bgr-flask-settings', JSON.stringify(this.settings));
    },

    resetSettings() {
      this.settings = Object.assign({}, defaults, { model_key: this.modelOptions?.[0] || defaults.model_key, model_dir: defaultModelDir || '' });
      this.persistSettings();
    },

    updateSelections(event) {
      const files = Array.from(event.target.files || []);
      this.selectedFiles = files;
    },

    handleDrop(event) {
      this.isDragging = false;
      const files = Array.from(event.dataTransfer?.files || []).filter((file) => file.type.startsWith('image/'));
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
      const [file] = event.target.files || [];
      this.batchFile = file || null;
      this.batchFileName = file ? file.name : '';
    },

    resetBatch() {
      this.batchFile = null;
      this.batchFileName = '';
      this.batchSummary = null;
      const input = document.getElementById('batch-input');
      if (input) {
        input.value = '';
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
      this.addActivity('info', `Starting processing for ${this.selectedFiles.length} image(s)…`);
      try {
        const response = await fetch('/process', {
          method: 'POST',
          body: form,
        });
        const payload = await response.json();
        if (!response.ok) {
          throw new Error(payload.error || 'Processing failed');
        }
        const results = payload.results || [];
        results.forEach((item) => {
          this.addActivity('success', `${item.result_name} processed successfully ✓`);
        });
        this.previewItems = [...results, ...this.previewItems].slice(0, 10);
        this.showToast('success', `Processed ${results.length} image(s) successfully.`);
        this.loadHistory();
      } catch (error) {
        console.error(error);
        this.addActivity('error', `Processing failed ✗ — Reason: ${error.message}`);
        this.showToast('error', error.message);
      } finally {
        this.isProcessing = false;
      }
    },

    async processBatch() {
      if (!this.batchFile) {
        this.showToast('error', 'Select a ZIP archive to process.');
        return;
      }
      const form = new FormData();
      form.append('archive', this.batchFile, this.batchFile.name);
      Object.entries(this.settings).forEach(([key, value]) => {
        if (typeof value === 'boolean') {
          form.append(key, value ? 'true' : 'false');
        } else {
          form.append(key, value ?? '');
        }
      });
      this.isBatchProcessing = true;
      this.addActivity('info', `Batch processing started for ${this.batchFile.name}…`);
      try {
        const response = await fetch('/batch', { method: 'POST', body: form });
        const payload = await response.json();
        if (!response.ok) {
          throw new Error(payload.error || 'Batch processing failed');
        }
        this.batchSummary = payload.summary ? { ...payload.summary, download_url: payload.download_url } : null;
        this.showToast('success', `Batch completed — ${payload.summary.success} succeeded, ${payload.summary.failed} failed.`);
        (payload.entries || []).forEach((entry) => {
          const message = entry.success
            ? `${entry.input} processed successfully ✓`
            : `${entry.input} failed ✗ — Reason: ${entry.error || 'Unknown error'}`;
          this.addActivity(entry.success ? 'success' : 'error', message);
        });
        this.loadHistory();
      } catch (error) {
        console.error(error);
        this.addActivity('error', `Batch failed ✗ — Reason: ${error.message}`);
        this.showToast('error', error.message);
      } finally {
        this.isBatchProcessing = false;
      }
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
      const id = `${Date.now()}-${Math.random().toString(16).slice(2)}`;
      const iconMap = {
        success: { icon: 'task_alt', iconClass: 'text-emerald-500' },
        error: { icon: 'error', iconClass: 'text-rose-500' },
        info: { icon: 'info', iconClass: 'text-slate-400' },
      };
      const meta = iconMap[level] || iconMap.info;
      this.activityLog = [{ id, message, icon: meta.icon, iconClass: meta.iconClass, class: level }, ...this.activityLog].slice(0, 50);
    },

    showToast(type, message) {
      const container = document.getElementById('toast-container');
      if (!container) {
        return;
      }
      const toast = document.createElement('div');
      toast.className = `toast ${type}`;
      toast.innerHTML = `
        <span class="material-symbols-rounded text-base">${type === 'success' ? 'check_circle' : type === 'error' ? 'error' : 'info'}</span>
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
}

window.appController = appController;
