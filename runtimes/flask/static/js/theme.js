/**
 * Theme management helpers for the Background Remover UI.
 *
 * Centralises the logic for reading and persisting the preferred colour
 * scheme so that both vanilla scripts and Alpine components can share a
 * consistent view of the current theme.
 */
const THEME_STORAGE_KEYS = ['bgr-theme', 'theme'];

/**
 * Read the stored theme preference, considering legacy key names.
 *
 * @returns {('light'|'dark'|null)} The stored preference, or null when absent.
 */
function getStoredThemePreference() {
  for (const key of THEME_STORAGE_KEYS) {
    const storedValue = localStorage.getItem(key);
    if (storedValue === 'light' || storedValue === 'dark') {
      return storedValue;
    }
  }
  return null;
}

/**
 * Persist the theme preference under each supported key.
 *
 * @param {('light'|'dark')} theme - The theme that should be stored.
 */
function persistThemePreference(theme) {
  for (const key of THEME_STORAGE_KEYS) {
    localStorage.setItem(key, theme);
  }
}

/**
 * Apply the initial theme preference using localStorage and system defaults.
 */
function applyInitialTheme() {
  const savedPreference = getStoredThemePreference();
  const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
  const themeToApply = savedPreference ?? (prefersDark ? 'dark' : 'light');
  const root = document.documentElement;
  root.classList.toggle('dark', themeToApply === 'dark');
  root.dataset.theme = themeToApply;
  root.style.colorScheme = themeToApply;
}

/**
 * Toggle between dark and light themes, persisting the choice.
 *
 * @returns {('light'|'dark')} The theme that is active after toggling.
 */
function toggleTheme() {
  const element = document.documentElement;
  const isDark = element.classList.toggle('dark');
  const theme = isDark ? 'dark' : 'light';
  element.dataset.theme = theme;
  element.style.colorScheme = theme;
  persistThemePreference(theme);
  return theme;
}

/**
 * Helper exposing the currently active theme mode.
 *
 * @returns {('light'|'dark')} The active theme.
 */
function currentTheme() {
  return document.documentElement.classList.contains('dark') ? 'dark' : 'light';
}

applyInitialTheme();
document.addEventListener('DOMContentLoaded', applyInitialTheme);

window.toggleTheme = toggleTheme;
window.currentTheme = currentTheme;
