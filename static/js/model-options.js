(function () {
  'use strict';

  /**
   * Build an Alpine.js data object that manages model-specific advanced options.
   * The state object keeps all reactive properties defined up front to avoid
   * undefined binding errors when Alpine initialises template expressions.
   *
   * @param {object} config - Payload passed from the template.
   * @returns {object} Alpine component state definition.
   */
  /**
   * Normalise configuration payloads originating from Alpine templates.
   * Accepts JSON strings, DOM elements carrying dataset metadata, or plain objects.
   *
   * @param {(string|Element|object|null)} config - Raw payload from ``x-data``.
   * @returns {object} Normalised configuration object.
   */
  function parseConfigPayload(config) {
    if (!config) {
      return {};
    }
    if (typeof config === 'string') {
      try {
        return JSON.parse(config);
      } catch (error) {
        console.warn('Failed to parse model options config payload.', error);
        return {};
      }
    }
    if (config instanceof Element) {
      var encoded = config.dataset ? config.dataset.modelOptionsConfig || config.dataset.modelOptions || '' : '';
      return parseConfigPayload(encoded);
    }
    if (typeof config === 'object') {
      return config;
    }
    return {};
  }

  function createModelOptionsState(config) {
    var safeConfig = parseConfigPayload(config);
    var initialValues = safeConfig.initialValues && typeof safeConfig.initialValues === 'object'
      ? safeConfig.initialValues
      : {};

    return {
      configs: safeConfig.configs && typeof safeConfig.configs === 'object' ? safeConfig.configs : {},
      lookup: safeConfig.lookup && typeof safeConfig.lookup === 'object' ? safeConfig.lookup : {},
      selectedModelKey: safeConfig.initialKey || '',
      selectedModelName: safeConfig.initialModelName || null,
      optionSchemas: {},
      hasAdvancedOptions: false,
      initialValues: initialValues,
      initialModelName: safeConfig.initialModelName || null,
      idPrefix: safeConfig.idPrefix || 'model',
      fieldValues: Object.assign({}, initialValues),

      init: function () {
        this.ensureSelectedModelKey();
        this.refreshModelContext();
        this.resetModelOptions(true);

        var self = this;
        this.$watch('selectedModelKey', function () {
          self.refreshModelContext();
          self.resetModelOptions(false);
        });
      },

      ensureSelectedModelKey: function () {
        if (this.selectedModelKey) {
          return;
        }
        var keys = Object.keys(this.lookup);
        if (keys.length > 0) {
          this.selectedModelKey = keys[0];
        }
      },

      refreshModelContext: function () {
        var key = this.selectedModelKey;
        var resolvedName = (this.lookup && this.lookup[key]) || key || null;
        this.selectedModelName = resolvedName;
        if (!this.initialModelName && resolvedName) {
          this.initialModelName = resolvedName;
        }
        var schemaMap = resolvedName && this.configs ? this.configs[resolvedName] : null;
        this.optionSchemas = schemaMap ? Object.assign({}, schemaMap) : {};
        this.hasAdvancedOptions = Object.keys(this.optionSchemas).length > 0;
      },

      inputType: function (schema) {
        if (!schema) {
          return 'text';
        }
        if (Array.isArray(schema.options) && schema.options.length > 0) {
          return 'select';
        }
        if (schema.type === 'int' || schema.type === 'float') {
          return 'number';
        }
        if (schema.type === 'bool') {
          return 'checkbox';
        }
        return 'text';
      },

      optionList: function (schema) {
        if (!schema || !Array.isArray(schema.options)) {
          return [];
        }
        return schema.options.map(function (entry) {
          if (entry && typeof entry === 'object' && Object.prototype.hasOwnProperty.call(entry, 'value')) {
            return { value: entry.value, label: entry.label || String(entry.value) };
          }
          return { value: entry, label: String(entry) };
        });
      },

      normaliseValue: function (value, schema) {
        if (!schema) {
          return value;
        }
        var type = schema.type;
        if (type === 'bool') {
          return Boolean(value);
        }
        if (type === 'int' || type === 'float') {
          if (value === undefined || value === null || value === '') {
            if (schema.default === undefined || schema.default === null) {
              return '';
            }
            return String(schema.default);
          }
          return String(value);
        }
        if (value === undefined || value === null) {
          return '';
        }
        return String(value);
      },

      numberStep: function (schema) {
        if (!schema) {
          return '1';
        }
        if (schema.step !== undefined && schema.step !== null && schema.step !== '') {
          return schema.step;
        }
        return schema.type === 'float' ? '0.1' : '1';
      },

      resetModelOptions: function (preserveExisting) {
        var schemas = this.optionSchemas || {};
        var keys = Object.keys(schemas);
        if (!keys.length) {
          this.fieldValues = {};
          return;
        }
        var nextValues = {};
        var reuseInitial = preserveExisting && this.initialModelName === this.selectedModelName;
        var self = this;
        keys.forEach(function (name) {
          var schema = schemas[name];
          var baseValue;
          if (preserveExisting && Object.prototype.hasOwnProperty.call(self.fieldValues, name)) {
            baseValue = self.fieldValues[name];
          } else if (
            reuseInitial &&
            Object.prototype.hasOwnProperty.call(self.initialValues || {}, name)
          ) {
            baseValue = self.initialValues[name];
          } else {
            baseValue = schema ? schema.default : undefined;
          }
          nextValues[name] = self.normaliseValue(baseValue, schema);
        });
        this.fieldValues = nextValues;
      },

      formatLabel: function (name, schema) {
        if (schema && schema.label) {
          return schema.label;
        }
        return String(name)
          .replace(/_/g, ' ')
          .replace(/\b\w/g, function (letter) {
            return letter.toUpperCase();
          });
      },
    };
  }

  document.addEventListener('alpine:init', function () {
    Alpine.data('modelOptionsForm', function (config) {
      return createModelOptionsState(config);
    });
  });

  if (!window.modelOptionsForm) {
    window.modelOptionsForm = function modelOptionsForm(config) {
      return createModelOptionsState(config);
    };
  }
})();
