/**
 * Settings Storage Utility
 * Manages persistent storage of user settings in localStorage
 * Provides fallback when backend is unavailable
 */

const STORAGE_KEY = 'neuraltrader_settings';

/**
 * Get settings from localStorage
 * @returns {Object|null} Settings object or null if not found
 */
export const getStoredSettings = () => {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    return stored ? JSON.parse(stored) : null;
  } catch (error) {
    console.error('Error reading settings from localStorage:', error);
    return null;
  }
};

/**
 * Save settings to localStorage
 * @param {Object} settings - Settings object to store
 * @returns {boolean} Success status
 */
export const saveStoredSettings = (settings) => {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
    return true;
  } catch (error) {
    console.error('Error saving settings to localStorage:', error);
    return false;
  }
};

/**
 * Clear settings from localStorage
 * @returns {boolean} Success status
 */
export const clearStoredSettings = () => {
  try {
    localStorage.removeItem(STORAGE_KEY);
    return true;
  } catch (error) {
    console.error('Error clearing settings from localStorage:', error);
    return false;
  }
};

/**
 * Get a specific setting value
 * @param {string} key - Setting key to retrieve
 * @param {*} defaultValue - Default value if not found
 * @returns {*} Setting value or default
 */
export const getSetting = (key, defaultValue = null) => {
  const settings = getStoredSettings();
  return settings && settings[key] !== undefined ? settings[key] : defaultValue;
};

/**
 * Update a specific setting
 * @param {string} key - Setting key to update
 * @param {*} value - New value
 * @returns {boolean} Success status
 */
export const updateSetting = (key, value) => {
  const settings = getStoredSettings() || {};
  settings[key] = value;
  return saveStoredSettings(settings);
};

/**
 * Check if settings exist in localStorage
 * @returns {boolean} True if settings exist
 */
export const hasStoredSettings = () => {
  return localStorage.getItem(STORAGE_KEY) !== null;
};

/**
 * Get default settings
 * @returns {Object} Default settings object
 */
export const getDefaultSettings = () => ({
  openai_api_key: "",
  gemini_api_key: "",
  selected_model: "gpt-4.1",
  selected_provider: "openai",
});

/**
 * Merge settings with defaults
 * @param {Object} settings - Settings to merge
 * @returns {Object} Merged settings
 */
export const mergeWithDefaults = (settings) => ({
  ...getDefaultSettings(),
  ...settings,
});

