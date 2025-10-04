import api from './api'

export interface Document {
  id: string
  filename: string
  file_path: string
  file_size: number
  status: 'pending' | 'processing' | 'completed' | 'failed'
  chunk_count: number
  created_at: string
  updated_at: string
  metadata?: Record<string, any>
}

export interface ProcessingStatus {
  total_documents: number
  processed_documents: number
  failed_documents: number
  processing_documents: number
  last_updated: string
}

export interface DocumentUploadResponse {
  message: string
  filename: string
  status: string
}

export const documentService = {
  // Get all documents with pagination and filtering
  async getDocuments(params?: {
    skip?: number
    limit?: number
    status?: string
  }): Promise<Document[]> {
    console.log('📄 Fetching documents with params:', params)
    try {
      const response = await api.get('/api/v1/documents', { params })
      console.log('✅ Documents fetched successfully:', response.data)
      return response.data
    } catch (error) {
      console.error('❌ Failed to fetch documents:', error)
      throw error
    }
  },

  // Get a specific document
  async getDocument(id: string): Promise<Document> {
    const response = await api.get(`/api/v1/documents/${id}`)
    return response.data
  },

  // Delete a document
  async deleteDocument(id: string): Promise<{ message: string }> {
    const response = await api.delete(`/api/v1/documents/${id}`)
    return response.data
  },

  // Upload a document
  async uploadDocument(file: File): Promise<DocumentUploadResponse> {
    const formData = new FormData()
    formData.append('file', file)
    
    const response = await api.post('/api/v1/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    })
    return response.data
  },

  // Get processing status
  async getProcessingStatus(): Promise<ProcessingStatus> {
    console.log('📊 Fetching processing status...')
    try {
      const response = await api.get('/api/v1/documents/status/summary')
      console.log('✅ Processing status fetched:', response.data)
      return response.data
    } catch (error) {
      console.error('❌ Failed to fetch processing status:', error)
      throw error
    }
  },

  // Reprocess all documents
  async reprocessDocuments(): Promise<{ message: string }> {
    const response = await api.post('/api/v1/admin/reprocess-documents')
    return response.data
  }
}
