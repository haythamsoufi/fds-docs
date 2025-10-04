import api from './api'

export interface QueryRequest {
  query: string
  max_results?: number
  include_metadata?: boolean
  filters?: Record<string, any>
}

export interface QueryResponse {
  query: string
  answer: string
  retrieved_chunks: RetrievedChunk[]
  response_time: number
  intent?: string
  confidence?: number
  citations?: Citation[]
}

export interface RetrievedChunk {
  chunk: {
    id: string
    document_id: string
    content: string
    metadata: Record<string, any>
  }
  score: number
  document: {
    id: string
    metadata: Record<string, any>
  }
}

export interface Citation {
  id: string
  document_id: string
  document_title?: string
  page_number?: number
  section_title?: string
  section_level?: number
  chunk_id: string
  score: number
  content: string
  metadata?: Record<string, any>
}

export interface QueryHistory {
  id: string
  query: string
  response_time: number
  timestamp: string
  results_count: number
}

export interface LLMStatus {
  status: 'available' | 'configured_but_unavailable' | 'not_configured' | 'error'
  openai_configured: boolean
  local_llm_configured: boolean
  local_llm_available: boolean
  response_mode: 'llm_generated' | 'extractive_summary'
  llm_model?: string
  base_url?: string
  error?: string
}

export const queryService = {
  // Search documents
  async searchDocuments(request: QueryRequest): Promise<QueryResponse> {
    const response = await api.post('/api/v1/query', request)
    return response.data
  },

  // Get query history
  async getQueryHistory(limit: number = 50): Promise<QueryHistory[]> {
    const response = await api.get('/api/v1/queries/history', {
      params: { limit }
    })
    return response.data
  },

  // Get LLM status
  async getLLMStatus(): Promise<LLMStatus> {
    const response = await api.get('/api/v1/llm/status')
    return response.data
  }
}
