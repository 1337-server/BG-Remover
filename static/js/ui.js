/* eslint-disable no-console */
(() => {
  'use strict';

  const storageKey = 'bg-remover-theme';
  const root = document.documentElement;

  const prefersDarkMediaQuery = typeof window.matchMedia === 'function'
    ? window.matchMedia('(prefers-color-scheme: dark)')
    : null;

  /**
   * Resolve the theme preference based on stored value or system preference.
   * @returns {('light'|'dark')} The preferred theme name.
   */
  const resolvePreferredTheme = () => {
    const stored = window.localStorage ? localStorage.getItem(storageKey) : null;
    if (stored === 'light' || stored === 'dark') {
      return stored;
    }
    if (prefersDarkMediaQuery && prefersDarkMediaQuery.matches) {
      return 'dark';
    }
    return 'light';
  };

  /**
   * Apply the requested theme to the root element and optionally persist it.
   * @param {('light'|'dark')} theme - Desired theme name.
   * @param {boolean} persist - Whether to store the preference.
   */
  const applyTheme = (theme, persist = true) => {
    const resolved = theme === 'dark' ? 'dark' : 'light';
    root.setAttribute('data-theme', resolved);
    root.classList.toggle('dark', resolved === 'dark');
    root.style.colorScheme = resolved;
    if (persist && window.localStorage) {
      localStorage.setItem(storageKey, resolved);
    }
    return resolved;
  };

  const themeToggle = document.querySelector('[data-theme-toggle]');
  const themeToggleLabel = themeToggle?.querySelector('.theme-toggle-label');
  const themeToggleIcon = themeToggle?.querySelector('.theme-toggle-icon');

  /**
   * Update the theme toggle button label and state.
   * @param {('light'|'dark')} theme - The currently active theme.
   */
  const syncThemeToggle = (theme) => {
    if (!themeToggle) {
      return;
    }
    const isDark = theme === 'dark';
    themeToggle.setAttribute('aria-pressed', String(isDark));
    if (themeToggleLabel) {
      themeToggleLabel.textContent = isDark ? 'Switch to light mode' : 'Switch to dark mode';
    }
    if (themeToggleIcon) {
      themeToggleIcon.innerHTML = isDark
        ? '<svg class="h-4 w-4" xmlns="http://www.w3.org/2000/svg" fill="currentColor" viewBox="0 0 24 24"><path d="M12 3a1 1 0 0 1 1 1v2a1 1 0 0 1-2 0V4a1 1 0 0 1 1-1Zm6.364 2.05a1 1 0 0 1 0 1.414l-1.415 1.414a1 1 0 0 1-1.414-1.414l1.414-1.414a1 1 0 0 1 1.415 0ZM12 7a5 5 0 1 1-5 5 5 5 0 0 1 5-5Zm9 4a1 1 0 0 1 0 2h-2a1 1 0 0 1 0-2h2Zm-3.636 6.95a1 1 0 0 1 0 1.414l-1.414 1.415a1 1 0 1 1-1.415-1.415l1.415-1.414a1 1 0 0 1 1.414 0ZM13 20v-2a1 1 0 0 0-2 0v2a1 1 0 0 0 2 0ZM6.05 5.05a1 1 0 0 1 1.414 0L8.88 6.464A1 1 0 0 1 7.464 7.88L6.05 6.465a1 1 0 0 1 0-1.414ZM6 13a1 1 0 0 1-1 1H3a1 1 0 0 1 0-2h2a1 1 0 0 1 1 1Zm-.95 4.95a1 1 0 0 1 1.414 0l1.415 1.414a1 1 0 0 1-1.415 1.415L5.05 18.364a1 1 0 0 1 0-1.414Z"/></svg>'
        : '<svg class="h-4 w-4" xmlns="http://www.w3.org/2000/svg" fill="currentColor" viewBox="0 0 24 24"><path d="M21 12.79A9 9 0 0 1 11.21 3 7 7 0 1 0 21 12.79Z"/></svg>';
    }
  };

  const currentTheme = applyTheme(resolvePreferredTheme(), false);
  syncThemeToggle(currentTheme);

  if (themeToggle) {
    themeToggle.addEventListener('click', () => {
      const nextTheme = root.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
      const applied = applyTheme(nextTheme, true);
      syncThemeToggle(applied);
    });
  }

  if (prefersDarkMediaQuery) {
    prefersDarkMediaQuery.addEventListener('change', (event) => {
      const stored = window.localStorage ? localStorage.getItem(storageKey) : null;
      if (stored !== 'light' && stored !== 'dark') {
        const applied = applyTheme(event.matches ? 'dark' : 'light', false);
        syncThemeToggle(applied);
      }
    });
  }

  if (window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
    document.body?.setAttribute('data-reduced-motion', 'true');
  }

  const toastRegion = document.getElementById('toast-region');
  const activeToasts = new Set();

  /**
   * Create and show a toast notification.
   * @param {string} message - The message to display.
   * @param {'info'|'success'|'error'} tone - Visual tone for the toast.
   */
  const showToast = (message, tone = 'info') => {
    if (!toastRegion || typeof message !== 'string' || !message.trim()) {
      return;
    }
    const toneClasses = {
      info: 'bg-[var(--color-surface-elevated)] text-[var(--color-text-muted)] border-[var(--color-border-muted)]',
      success: 'bg-emerald-500 text-white border-transparent',
      error: 'bg-rose-500 text-white border-transparent',
    };
    const toast = document.createElement('div');
    toast.className = `pointer-events-auto rounded-2xl border px-4 py-3 text-sm shadow-lg transition ${toneClasses[tone] || toneClasses.info}`;
    toast.setAttribute('role', 'status');
    toast.textContent = message.trim();
    toastRegion.appendChild(toast);
    activeToasts.add(toast);

    window.setTimeout(() => {
      toast.classList.add('opacity-0', 'translate-y-1');
      window.setTimeout(() => {
        activeToasts.delete(toast);
        toast.remove();
      }, 220);
    }, 4200);
  };

  const openModal = (modal) => {
    if (!modal) {
      return;
    }
    modal.classList.remove('hidden');
    modal.classList.add('flex');
    modal.classList.remove('pointer-events-none');
    const panel = modal.querySelector('[data-help-modal-panel]');
    const focusable = panel?.querySelector('button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])');
    if (focusable instanceof HTMLElement) {
      window.requestAnimationFrame(() => focusable.focus());
    }
  };

  const closeModal = (modal) => {
    if (!modal) {
      return;
    }
    modal.classList.add('hidden');
    modal.classList.remove('flex');
    modal.classList.add('pointer-events-none');
  };

  const helpModal = document.getElementById('help-modal');
  if (helpModal) {
    document.querySelectorAll('[data-help-modal-open]').forEach((trigger) => {
      trigger.addEventListener('click', () => {
        openModal(helpModal);
        trigger.setAttribute('aria-expanded', 'true');
      });
    });

    const closeButtons = helpModal.querySelectorAll('[data-help-modal-close]');
    closeButtons.forEach((button) => {
      button.addEventListener('click', () => {
        closeModal(helpModal);
        document.querySelectorAll('[data-help-modal-open]').forEach((trigger) => {
          trigger.setAttribute('aria-expanded', 'false');
        });
      });
    });

    helpModal.addEventListener('click', (event) => {
      if (event.target === helpModal) {
        closeModal(helpModal);
        document.querySelectorAll('[data-help-modal-open]').forEach((trigger) => {
          trigger.setAttribute('aria-expanded', 'false');
        });
      }
    });

    document.addEventListener('keydown', (event) => {
      if (event.key === 'Escape' && !helpModal.classList.contains('hidden')) {
        closeModal(helpModal);
        document.querySelectorAll('[data-help-modal-open]').forEach((trigger) => {
          trigger.setAttribute('aria-expanded', 'false');
        });
      }
    });
  }

  const tooltipState = new Map();

  const hideTooltip = (id) => {
    const state = tooltipState.get(id);
    if (!state) {
      return;
    }
    state.panel.classList.add('hidden');
    state.panel.setAttribute('data-visible', 'false');
    state.trigger.setAttribute('aria-expanded', 'false');
  };

  const showTooltip = (id) => {
    const state = tooltipState.get(id);
    if (!state) {
      return;
    }
    tooltipState.forEach((value, key) => {
      if (key !== id) {
        hideTooltip(key);
      }
    });
    state.panel.classList.remove('hidden');
    state.panel.setAttribute('data-visible', 'true');
    state.trigger.setAttribute('aria-expanded', 'true');
  };

  document.querySelectorAll('[data-tooltip-trigger]').forEach((trigger) => {
    const tooltipId = trigger.getAttribute('data-tooltip-id');
    if (!tooltipId) {
      return;
    }
    const panel = document.getElementById(tooltipId);
    if (!panel) {
      return;
    }
    tooltipState.set(tooltipId, { trigger, panel });

    trigger.addEventListener('mouseenter', () => showTooltip(tooltipId));
    trigger.addEventListener('focus', () => showTooltip(tooltipId));
    trigger.addEventListener('click', () => {
      const isVisible = panel.getAttribute('data-visible') === 'true';
      if (isVisible) {
        hideTooltip(tooltipId);
      } else {
        showTooltip(tooltipId);
      }
    });
    trigger.addEventListener('mouseleave', () => hideTooltip(tooltipId));
    trigger.addEventListener('blur', () => hideTooltip(tooltipId));

    panel.addEventListener('mouseenter', () => showTooltip(tooltipId));
    panel.addEventListener('mouseleave', () => hideTooltip(tooltipId));
  });

  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') {
      tooltipState.forEach((_, id) => hideTooltip(id));
    }
  });

  document.querySelectorAll('[data-help-toggle]').forEach((toggleWrapper) => {
    const input = toggleWrapper.querySelector('input[type="checkbox"]');
    const targetSelector = toggleWrapper.getAttribute('data-help-target');
    const target = targetSelector ? document.querySelector(targetSelector) : toggleWrapper.closest('[data-help-container]');

    if (!(input instanceof HTMLInputElement) || !target) {
      return;
    }

    const applyState = () => {
      target.classList.toggle('help-hints-hidden', !input.checked);
    };

    applyState();
    input.addEventListener('change', applyState);
  });

  document.querySelectorAll('[data-tab-group]').forEach((group) => {
    const triggers = Array.from(group.querySelectorAll('[data-tab-target]'));
    const panels = Array.from(group.querySelectorAll('[data-tab-panel]'));

    const activateTab = (target) => {
      triggers.forEach((button) => {
        const isActive = button.getAttribute('data-tab-target') === target;
        button.setAttribute('aria-selected', String(isActive));
        button.classList.toggle('bg-[var(--color-accent-soft)]', isActive);
        button.classList.toggle('text-[var(--color-text-strong)]', isActive);
        button.classList.toggle('text-[var(--color-text-muted)]', !isActive);
      });
      panels.forEach((panel) => {
        const isActive = panel.getAttribute('data-tab-panel') === target;
        panel.classList.toggle('hidden', !isActive);
        panel.setAttribute('tabindex', isActive ? '0' : '-1');
        if (isActive) {
          panel.removeAttribute('hidden');
        } else {
          panel.setAttribute('hidden', '');
        }
      });
    };

    triggers.forEach((button) => {
      button.addEventListener('click', () => {
        activateTab(button.getAttribute('data-tab-target'));
      });
    });

    const initiallyActive = triggers.find((button) => button.getAttribute('aria-selected') === 'true');
    activateTab(initiallyActive?.getAttribute('data-tab-target') || (triggers[0]?.getAttribute('data-tab-target') ?? ''));
  });

  const updateDropzoneInfo = (input, infoElement, previewElement) => {
    if (!(input instanceof HTMLInputElement)) {
      return;
    }
    const file = input.files && input.files[0];
    if (file) {
      if (infoElement) {
        const sizeInKb = file.size / 1024;
        const sizeLabel = sizeInKb > 1024
          ? `${(sizeInKb / 1024).toFixed(2)} MB`
          : `${sizeInKb.toFixed(1)} KB`;
        infoElement.textContent = `${file.name} • ${sizeLabel}`;
      }
      if (previewElement instanceof HTMLImageElement) {
        const reader = new FileReader();
        reader.onload = () => {
          if (typeof reader.result === 'string') {
            previewElement.src = reader.result;
            previewElement.classList.remove('hidden');
          }
        };
        reader.readAsDataURL(file);
      }
    } else {
      if (infoElement) {
        infoElement.textContent = 'No file selected yet.';
      }
      if (previewElement instanceof HTMLImageElement) {
        previewElement.classList.add('hidden');
        previewElement.removeAttribute('src');
      }
    }
  };

  document.querySelectorAll('[data-dropzone]').forEach((dropzone) => {
    const inputSelector = dropzone.getAttribute('data-dropzone-input');
    const infoSelector = dropzone.getAttribute('data-dropzone-info');
    const previewSelector = dropzone.getAttribute('data-dropzone-preview');
    const input = inputSelector ? dropzone.querySelector(inputSelector) || document.querySelector(inputSelector) : dropzone.querySelector('input[type="file"]');
    const infoElement = infoSelector ? document.querySelector(infoSelector) : dropzone.querySelector('[data-dropzone-info]');
    const previewElement = previewSelector ? document.querySelector(previewSelector) : dropzone.querySelector('img');

    if (!(input instanceof HTMLInputElement)) {
      return;
    }

    dropzone.addEventListener('click', () => {
      input.click();
    });

    input.addEventListener('change', () => {
      updateDropzoneInfo(input, infoElement, previewElement);
    });

    dropzone.addEventListener('dragover', (event) => {
      event.preventDefault();
      dropzone.classList.add('ring-2', 'ring-[var(--color-accent-strong)]');
    });

    dropzone.addEventListener('dragleave', () => {
      dropzone.classList.remove('ring-2', 'ring-[var(--color-accent-strong)]');
    });

    dropzone.addEventListener('drop', (event) => {
      event.preventDefault();
      dropzone.classList.remove('ring-2', 'ring-[var(--color-accent-strong)]');
      if (event.dataTransfer?.files?.length) {
        input.files = event.dataTransfer.files;
        input.dispatchEvent(new Event('change', { bubbles: true }));
      }
    });

    updateDropzoneInfo(input, infoElement, previewElement);
  });

  const FORMAT_EXTENSION_MAP = new Map([
    ['png', '.png'],
    ['webp', '.webp'],
    ['jpg', '.jpg'],
    ['jpeg', '.jpg'],
  ]);

  let downloadObjectUrl = null;
  let activeFilename = 'image';

  const normaliseFormatKey = (value) => {
    if (typeof value !== 'string') {
      return '';
    }
    return value.trim().toLowerCase();
  };

  const inferExtension = (formatKey, mimeType) => {
    const normalisedKey = normaliseFormatKey(formatKey);
    if (FORMAT_EXTENSION_MAP.has(normalisedKey)) {
      return FORMAT_EXTENSION_MAP.get(normalisedKey);
    }
    if (typeof mimeType === 'string' && mimeType.includes('/')) {
      const subtype = mimeType.split('/').pop();
      if (subtype) {
        return `.${subtype.toLowerCase()}`;
      }
    }
    return '';
  };

  const ensureExtension = (name, extension) => {
    if (!name) {
      return '';
    }
    if (!extension) {
      return name;
    }
    const normalisedName = name.toLowerCase();
    if (normalisedName.endsWith(extension.toLowerCase())) {
      return name;
    }
    const stripped = name.replace(/\.[^./\\\s]+$/, '');
    return `${stripped}${extension}`;
  };

  const resolveDownloadName = (preferredName, formatKey, mimeType) => {
    const extension = inferExtension(formatKey, mimeType);
    if (typeof preferredName === 'string' && preferredName.trim()) {
      return ensureExtension(preferredName.trim(), extension);
    }
    if (typeof activeFilename === 'string' && activeFilename.trim()) {
      const safeName = activeFilename.trim().replace(/[^A-Za-z0-9._-]/g, '_');
      return ensureExtension(safeName, extension);
    }
    return '';
  };

  const formatDownloadLabel = (preferredName, formatKey, mimeType) => {
    const resolvedName = (preferredName && preferredName.trim()) || activeFilename || 'result';
    const normalisedKey = normaliseFormatKey(formatKey).toUpperCase();
    if (normalisedKey) {
      return `Download ${resolvedName} (${normalisedKey})`;
    }
    if (typeof mimeType === 'string' && mimeType.includes('/')) {
      const subtype = mimeType.split('/').pop();
      if (subtype) {
        return `Download ${resolvedName} (${subtype.toUpperCase()})`;
      }
    }
    return `Download ${resolvedName}`;
  };

  const showSpinner = (spinner, shouldShow) => {
    if (!spinner) {
      return;
    }
    spinner.classList.toggle('hidden', !shouldShow);
    spinner.setAttribute('aria-hidden', shouldShow ? 'false' : 'true');
  };

  const createAlertManager = (element) => ({
    show(message) {
      if (!element) {
        return;
      }
      element.textContent = message;
      element.classList.remove('hidden');
      if (typeof element.focus === 'function') {
        element.focus();
      }
    },
    clear() {
      if (!element) {
        return;
      }
      element.textContent = '';
      element.classList.add('hidden');
    },
  });

  const normaliseImagePayload = (value, fallbackMimeType) => {
    if (typeof value !== 'string' || !value.trim()) {
      return { base64: null, dataUrl: null, mimeType: fallbackMimeType || null };
    }
    const trimmed = value.trim();
    if (trimmed.startsWith('data:')) {
      const match = trimmed.match(/^data:([^;]+);base64,(.+)$/);
      if (match) {
        return { base64: match[2], dataUrl: trimmed, mimeType: match[1] || fallbackMimeType || null };
      }
      return { base64: null, dataUrl: trimmed, mimeType: fallbackMimeType || null };
    }
    const mimeType = fallbackMimeType || 'image/png';
    return {
      base64: trimmed,
      dataUrl: `data:${mimeType};base64,${trimmed}`,
      mimeType,
    };
  };

  const base64ToBlob = (base64Payload, mimeType) => {
    if (typeof base64Payload !== 'string' || !base64Payload) {
      throw new Error('No image data available to download.');
    }
    const binaryString = window.atob(base64Payload);
    const length = binaryString.length;
    const bytes = new Uint8Array(length);
    for (let index = 0; index < length; index += 1) {
      bytes[index] = binaryString.charCodeAt(index);
    }
    return new Blob([bytes], { type: mimeType || 'application/octet-stream' });
  };

  const clampInputValue = (input) => {
    if (!(input instanceof HTMLInputElement)) {
      return;
    }

    const rawValue = input.value;
    if (rawValue === '') {
      return;
    }

    const numericValue = Number(rawValue);
    if (Number.isNaN(numericValue)) {
      return;
    }

    const hasMin = input.min !== '';
    const hasMax = input.max !== '';
    const minValue = hasMin ? Number(input.min) : Number.NEGATIVE_INFINITY;
    const maxValue = hasMax ? Number(input.max) : Number.POSITIVE_INFINITY;
    let clampedValue = numericValue;

    if (!Number.isNaN(minValue) && numericValue < minValue) {
      clampedValue = minValue;
    }
    if (!Number.isNaN(maxValue) && clampedValue > maxValue) {
      clampedValue = maxValue;
    }

    if (clampedValue !== numericValue) {
      input.value = clampedValue.toString();
    }
  };

  const enforceNumericBounds = () => {
    const numericInputs = document.querySelectorAll(
      '#single-form input[type="number"], #single-form input[type="range"], ' +
        '#folder-form input[type="number"], #folder-form input[type="range"]',
    );

    numericInputs.forEach((input) => {
      clampInputValue(input);
      input.addEventListener('change', () => clampInputValue(input));
      input.addEventListener('blur', () => clampInputValue(input));
    });
  };

  enforceNumericBounds();

  const acceleratorRuntimeBadge = document.getElementById('accelerator-runtime-badge');
  const singleForm = document.getElementById('single-form');
  const singleFileInput = document.getElementById('single-file');
  const singleSpinner = document.getElementById('single-spinner');
  const singleResult = document.getElementById('single-result');
  const singlePreview = document.getElementById('single-preview');
  const singleDownload = document.getElementById('single-download');
  const singleSummary = document.getElementById('single-summary');
  const singleStatusText = document.getElementById('single-status-text');
  const singleFormatSelect = document.getElementById('single-format');
  const singleAlert = document.getElementById('single-form-alert');
  const singleAlertManager = createAlertManager(singleAlert);
  const singleCopyButton = document.getElementById('single-copy');
  const folderForm = document.getElementById('folder-form');

  if (singleCopyButton && !navigator.clipboard) {
    singleCopyButton.classList.add('hidden');
    singleCopyButton.setAttribute('hidden', '');
  }

  const updateAcceleratorUi = (state = {}) => {
    const message = (state.accelerator_message || 'Using CPU (CPUExecutionProvider)').toString();
    if (acceleratorRuntimeBadge) {
      acceleratorRuntimeBadge.textContent = message;
    }
  };

  const previewSizeInput = document.getElementById('single-preview-size');
  const previewSizeLabel = document.getElementById('single-preview-size-label');
  const previewContainer = document.getElementById('single-preview')?.parentElement;

  const applyPreviewSize = () => {
    if (!previewSizeInput) {
      return;
    }
    const value = Number(previewSizeInput.value) || 0;
    if (previewSizeLabel) {
      previewSizeLabel.textContent = value ? `${value} px` : 'Auto';
    }
    const sizeText = value > 0 ? `${value}px` : '';
    if (previewContainer) {
      previewContainer.style.maxWidth = sizeText || '100%';
    }
    if (singlePreview) {
      singlePreview.style.maxWidth = sizeText || '100%';
    }
  };

  if (previewSizeInput) {
    previewSizeInput.addEventListener('input', applyPreviewSize);
    previewSizeInput.addEventListener('change', applyPreviewSize);
    applyPreviewSize();
  }

  const edgeSlider = document.getElementById('single-feather');
  const edgeValueLabel = document.getElementById('single-feather-value');

  const syncEdgeValue = () => {
    if (edgeSlider && edgeValueLabel) {
      edgeValueLabel.textContent = edgeSlider.value;
    }
  };

  if (edgeSlider) {
    edgeSlider.addEventListener('input', syncEdgeValue);
    edgeSlider.addEventListener('change', syncEdgeValue);
    syncEdgeValue();
  }

  const revokeDownloadUrl = () => {
    if (downloadObjectUrl) {
      URL.revokeObjectURL(downloadObjectUrl);
      downloadObjectUrl = null;
    }
  };

  const clearDownloadLink = () => {
    if (!singleDownload) {
      return;
    }
    revokeDownloadUrl();
    singleDownload.classList.add('hidden');
    singleDownload.removeAttribute('href');
    singleDownload.removeAttribute('download');
    const defaultFormat = singleFormatSelect && singleFormatSelect.value;
    singleDownload.textContent = formatDownloadLabel(null, defaultFormat, null);
  };

  clearDownloadLink();

  if (singleFormatSelect && singleDownload) {
    singleFormatSelect.addEventListener('change', () => {
      if (singleDownload.classList.contains('hidden')) {
        singleDownload.textContent = formatDownloadLabel(null, singleFormatSelect.value, null);
      }
    });
  }

  const describeStatus = (result) => {
    if (!result) {
      return 'Background removal completed successfully.';
    }
    if (result.success) {
      return result.error
        ? `Completed with notes: ${result.error}`
        : 'Background removal completed successfully.';
    }
    return result.error ? `Error: ${result.error}` : 'Background removal failed.';
  };

  if (singleCopyButton && singlePreview) {
    singleCopyButton.addEventListener('click', async () => {
      if (!singlePreview.src) {
        showToast('No preview available yet.', 'info');
        return;
      }
      try {
        await navigator.clipboard.writeText(singlePreview.src);
        showToast('Preview URL copied to clipboard.', 'success');
      } catch (error) {
        console.error('Unable to copy preview URL', error);
        showToast('Copy failed. Try again after interacting with the page.', 'error');
      }
    });
  }

  const previewToastMessage = 'Preview ready — background removed successfully.';

  if (singleForm) {
    singleForm.addEventListener('submit', async (event) => {
      event.preventDefault();

      if (!singleForm.checkValidity()) {
        if (typeof singleForm.reportValidity === 'function') {
          singleForm.reportValidity();
        }
        singleAlertManager.show('Please fix the highlighted errors before submitting the form.');
        return;
      }

      const file = singleFileInput?.files?.[0];
      if (!file) {
        singleAlertManager.show('Please choose an image to upload.');
        return;
      }

      singleAlertManager.clear();
      activeFilename = file.name || 'image';

      singleResult?.setAttribute('hidden', '');
      if (singlePreview) {
        singlePreview.removeAttribute('src');
      }
      if (singleStatusText) {
        singleStatusText.textContent = '';
      }
      if (singleSummary) {
        singleSummary.textContent = '';
      }
      clearDownloadLink();

      const formData = new FormData(singleForm);
      showSpinner(singleSpinner, true);

      try {
        const response = await fetch('/image/remove-bg?json=1', {
          method: 'POST',
          body: formData,
        });

        const payload = await response.json().catch(() => null);
        if (!response.ok || !payload) {
          const message = payload?.error || 'Background removal failed.';
          throw new Error(message);
        }

        updateAcceleratorUi(payload);

        const mimeType = payload.mime_type || 'image/png';
        const imagePayload = normaliseImagePayload(payload.image_base64, mimeType);
        if (singlePreview) {
          if (imagePayload.dataUrl) {
            singlePreview.src = imagePayload.dataUrl;
          } else {
            singlePreview.removeAttribute('src');
          }
        }
        applyPreviewSize();

        if (singleResult) {
          singleResult.removeAttribute('hidden');
        }
        if (singleStatusText) {
          singleStatusText.textContent = describeStatus(payload.result);
        }
        if (singleSummary) {
          if (payload.result) {
            const summaryData = { ...payload.result };
            if (payload.selection) {
              summaryData.selection = payload.selection;
            }
            singleSummary.textContent = JSON.stringify(summaryData, null, 2);
          } else {
            singleSummary.textContent = 'Background removal completed successfully.';
          }
        }

        if (imagePayload.dataUrl) {
          showToast(previewToastMessage, 'success');
        }

        if (imagePayload.base64 && singleDownload) {
          try {
            const blob = base64ToBlob(imagePayload.base64, imagePayload.mimeType || mimeType);
            revokeDownloadUrl();
            downloadObjectUrl = URL.createObjectURL(blob);
            const resolvedDownloadName = resolveDownloadName(
              payload.download_name,
              payload.format,
              imagePayload.mimeType || mimeType,
            );
            singleDownload.href = downloadObjectUrl;
            if (resolvedDownloadName) {
              singleDownload.download = resolvedDownloadName;
            } else {
              singleDownload.removeAttribute('download');
            }
            singleDownload.textContent = formatDownloadLabel(
              resolvedDownloadName,
              payload.format,
              imagePayload.mimeType || mimeType,
            );
            singleDownload.classList.remove('hidden');
          } catch (error) {
            console.error('Unable to prepare the download link.', error);
            clearDownloadLink();
          }
        } else {
          clearDownloadLink();
        }
      } catch (error) {
        singleAlertManager.show(error.message || 'Background removal failed.');
      } finally {
        showSpinner(singleSpinner, false);
      }
    });
  }

  const folderSpinner = document.getElementById('folder-spinner');
  const folderResult = document.getElementById('folder-result');
  const folderSummary = document.getElementById('folder-summary');
  const folderTableBody = document.querySelector('#folder-table tbody');
  const folderZipLink = document.getElementById('folder-zip-link');
  const folderZipButton = document.getElementById('folder-zip-button');
  const folderAlert = document.getElementById('folder-form-alert');
  const folderAlertManager = createAlertManager(folderAlert);

  if (folderForm) {
    folderForm.addEventListener('submit', async (event) => {
      event.preventDefault();
      if (!folderForm.checkValidity()) {
        if (typeof folderForm.reportValidity === 'function') {
          folderForm.reportValidity();
        }
        folderAlertManager.show('Please fix the highlighted errors before submitting the form.');
        return;
      }

      folderAlertManager.clear();

      const formData = new FormData(folderForm);
      let endpoint = '/image/remove-bg?json=1';
      const zipToggle = document.getElementById('folder-zip');
      if (zipToggle instanceof HTMLInputElement && zipToggle.checked) {
        endpoint = '/image/remove-bg?json=1&zip=1';
      }

      showSpinner(folderSpinner, true);
      try {
        const response = await fetch(endpoint, {
          method: 'POST',
          body: formData,
        });
        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.error || 'Folder processing failed.');
        }

        updateAcceleratorUi(data);

        if (folderResult) {
          folderResult.removeAttribute('hidden');
        }
        if (folderSummary) {
          folderSummary.classList.remove('hidden');
          const summarySuccess = Number(data?.summary?.success) || 0;
          const summaryTotal = Number(data?.summary?.total) || 0;
          let summaryMessage = `Processed ${summarySuccess} of ${summaryTotal} images.`;
          if (data?.selection) {
            const modelLabel = data.selection.removal_model_label || '';
            const modelName = data.selection.model_name || '';
            const modelDescription = modelLabel && modelName && modelLabel !== modelName
              ? `${modelLabel} (${modelName})`
              : (modelLabel || modelName);
            if (modelDescription) {
              summaryMessage += ` Using ${modelDescription}.`;
            }
          }
          folderSummary.textContent = summaryMessage;
          if (data.zip_error) {
            folderSummary.textContent += ` ZIP error: ${data.zip_error}`;
          }
        }

        if (folderTableBody) {
          folderTableBody.innerHTML = '';
          data.results.forEach((item) => {
            const row = document.createElement('tr');

            const fileCell = document.createElement('td');
            fileCell.className = 'px-4 py-3 align-top text-[var(--color-text-muted)]';
            fileCell.textContent = item.path_in;
            row.appendChild(fileCell);

            const previewCell = document.createElement('td');
            previewCell.className = 'px-4 py-3 align-top';
            if (item.preview_url) {
              const previewImage = document.createElement('img');
              previewImage.src = item.preview_url;
              previewImage.alt = `Preview of ${item.path_in}`;
              previewImage.className = 'h-16 w-16 rounded-xl object-cover shadow';
              previewCell.appendChild(previewImage);
            } else {
              previewCell.textContent = item.error ? 'No preview' : '-';
            }
            row.appendChild(previewCell);

            const statusCell = document.createElement('td');
            statusCell.className = 'px-4 py-3 align-top font-semibold';
            statusCell.textContent = item.success ? 'Success' : 'Failed';
            statusCell.classList.toggle('text-emerald-500', Boolean(item.success));
            statusCell.classList.toggle('text-rose-500', !item.success);
            row.appendChild(statusCell);

            const timeCell = document.createElement('td');
            timeCell.className = 'px-4 py-3 align-top text-[var(--color-text-muted)]';
            const numericTime = Number(item.timing_ms);
            timeCell.textContent = Number.isFinite(numericTime)
              ? numericTime.toFixed(1)
              : (item.timing_ms ?? '-');
            row.appendChild(timeCell);

            const downloadCell = document.createElement('td');
            downloadCell.className = 'px-4 py-3 align-top';
            if (item.download_url) {
              const link = document.createElement('a');
              link.href = item.download_url;
              const formatLabel = (item.format || '').toString().toUpperCase();
              link.textContent = formatLabel ? `Download ${formatLabel}` : 'Download';
              link.className = 'inline-flex items-center rounded-full border border-[var(--color-border-muted)] px-3 py-1 text-xs font-medium text-[var(--color-text-muted)] transition hover:text-[var(--color-text-strong)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--color-accent-strong)] focus-visible:ring-offset-2 focus-visible:ring-offset-[var(--color-surface-elevated)]';
              downloadCell.appendChild(link);
            } else {
              downloadCell.textContent = item.error || '-';
            }
            row.appendChild(downloadCell);

            folderTableBody.appendChild(row);
          });
        }

        if (folderZipLink && folderZipButton) {
          if (data.zip_download_url) {
            folderZipLink.classList.remove('hidden');
            folderZipLink.removeAttribute('hidden');
            folderZipButton.href = data.zip_download_url;
            folderZipButton.download = 'results.zip';
          } else {
            folderZipLink.classList.add('hidden');
            folderZipLink.setAttribute('hidden', '');
            folderZipButton.removeAttribute('href');
          }
        }

        showToast('Folder processing completed.', 'success');
        folderAlertManager.clear();
      } catch (error) {
        folderAlertManager.show(error.message || 'Folder processing failed.');
      } finally {
        showSpinner(folderSpinner, false);
      }
    });
  }

  window.addEventListener('beforeunload', () => {
    revokeDownloadUrl();
    activeToasts.forEach((toast) => toast.remove());
  });
})();
