/**
 * API Configuration
 * Centralized API URL management with fallback to localhost
 */

// Get backend URL from environment variable or use localhost as fallback
const getBackendURL = () => {
  const envURL = process.env.REACT_APP_BACKEND_URL;
  
  // If environment variable is set and not empty, use it
  if (envURL && envURL.trim() !== '') {
    return envURL;
  }
  
  // Default to localhost for local development
  return 'http://localhost:8005';
};

export const BACKEND_URL = getBackendURL();
export const API_URL = `${BACKEND_URL}/api`;

// Log the configuration in development
if (process.env.NODE_ENV === 'development') {
  console.log('🔧 API Configuration:', {
    BACKEND_URL,
    API_URL,
    envVariable: process.env.REACT_APP_BACKEND_URL || '(not set - using default)'
  });
}

