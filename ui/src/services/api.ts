import axios from 'axios'

const API_BASE_URL = (import.meta as any).env.VITE_API_URL || 'http://localhost:8080'

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 0, // no timeout; user prefers waiting for best answer
  headers: {
    'Content-Type': 'application/json',
  },
})

// Request interceptor
api.interceptors.request.use(
  (config) => {
    console.log(`🚀 API Request: ${config.method?.toUpperCase()} ${config.url}`)
    console.log('📤 Request data:', config.data)
    console.log('🔗 Full URL:', `${config.baseURL}${config.url}`)
    
    // Add auth token if available
    const token = localStorage.getItem('auth_token')
    if (token) {
      config.headers.Authorization = `Bearer ${token}`
    }
    return config
  },
  (error) => {
    console.error('❌ Request error:', error)
    return Promise.reject(error)
  }
)

// Response interceptor
api.interceptors.response.use(
  (response) => {
    console.log(`✅ API Response: ${response.status} ${response.config.url}`)
    console.log('📥 Response data:', response.data)
    return response
  },
  (error) => {
    const errorDetails = {
      message: error.message,
      status: error.response?.status,
      statusText: error.response?.statusText,
      url: error.config?.url,
      method: error.config?.method,
      data: error.response?.data,
      baseURL: error.config?.baseURL
    }
    
    console.error('❌ API Error:', errorDetails)
    
    // Provide more specific error messages based on status codes
    if (error.response?.status === 401) {
      // Handle unauthorized access
      localStorage.removeItem('auth_token')
      window.location.href = '/login'
    } else if (error.response?.status === 404) {
      console.error('❌ Endpoint not found:', error.config?.url)
    } else if (error.response?.status >= 500) {
      console.error('❌ Server error:', error.response?.data?.detail || 'Internal server error')
    } else if (error.code === 'NETWORK_ERROR' || error.message === 'Network Error') {
      console.error('❌ Network error - check if backend is running')
    }
    
    return Promise.reject(error)
  }
)

export const monitoringService = {
  getHealth: () => api.get('/api/v1/monitoring/health'),
  getMetrics: () => api.get('/metrics')
};

export default api
