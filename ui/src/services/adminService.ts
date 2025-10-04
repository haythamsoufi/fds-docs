import api from './api'

export interface SystemStats {
  documents: {
    total: number
    processed: number
    failed: number
    processing: number
    pending: number
  }
  chunks: {
    total: number
    with_embeddings: number
    embedding_coverage: number
  }
  queries: {
    total: number
    avg_response_time: number
  }
  cache: {
    hit_rate: number
  }
}

export interface HealthCheck {
  status: 'healthy' | 'degraded' | 'unhealthy'
  timestamp: string
  version: string
  components: Record<string, string>
}

export const adminService = {
  // Get system statistics
  async getSystemStats(): Promise<SystemStats> {
    const response = await api.get('/api/v1/admin/stats')
    return response.data
  },

  // Get health check
  async getHealthCheck(): Promise<HealthCheck> {
    const response = await api.get('/health')
    return response.data
  },

  // Clear cache
  async clearCache(): Promise<{ message: string }> {
    const response = await api.post('/api/v1/admin/clear-cache')
    return response.data
  },

  // Reprocess documents
  async reprocessDocuments(): Promise<{ message: string }> {
    const response = await api.post('/api/v1/admin/reprocess-documents')
    return response.data
  },

  // Populate vector store
  async populateVectors(): Promise<{ message: string; status: string }> {
    const response = await api.post('/api/v1/admin/populate-vectors')
    return response.data
  }
}
