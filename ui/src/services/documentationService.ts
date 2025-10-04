import axios from 'axios'

export interface DocumentationFile {
  filename: string
  title: string
  description: string
  category: string
  size?: number
  last_modified?: string
}

export interface DocumentationList {
  files: DocumentationFile[]
}

export interface DocumentationCategory {
  id: string
  name: string
  description: string
  count: number
}

export interface DocumentationHealth {
  status: string
  message: string
  docs_directory: string
  files_count?: number
  available_files?: string[]
}

class DocumentationService {
  private baseURL = '/api/v1/docs'

  /**
   * List all available documentation files
   */
  async listDocumentation(): Promise<DocumentationList> {
    try {
      const response = await axios.get<DocumentationList>(`${this.baseURL}/`)
      return response.data
    } catch (error) {
      console.error('Error listing documentation:', error)
      throw new Error('Failed to list documentation files')
    }
  }

  /**
   * Get a specific documentation file content
   */
  async getDocumentation(filename: string): Promise<string> {
    try {
      const response = await axios.get(`${this.baseURL}/${filename}/content`, {
        responseType: 'text'
      })
      return response.data
    } catch (error) {
      console.error(`Error getting documentation ${filename}:`, error)
      throw new Error(`Failed to load documentation: ${filename}`)
    }
  }

  /**
   * Download a documentation file
   */
  async downloadDocumentation(filename: string): Promise<void> {
    try {
      const response = await axios.get(`${this.baseURL}/${filename}`, {
        responseType: 'blob'
      })
      
      // Create download link
      const url = window.URL.createObjectURL(new Blob([response.data]))
      const link = document.createElement('a')
      link.href = url
      link.setAttribute('download', filename)
      document.body.appendChild(link)
      link.click()
      link.remove()
      window.URL.revokeObjectURL(url)
    } catch (error) {
      console.error(`Error downloading documentation ${filename}:`, error)
      throw new Error(`Failed to download documentation: ${filename}`)
    }
  }

  /**
   * List documentation categories
   */
  async listCategories(): Promise<{ categories: DocumentationCategory[] }> {
    try {
      const response = await axios.get(`${this.baseURL}/categories/list`)
      return response.data
    } catch (error) {
      console.error('Error listing categories:', error)
      throw new Error('Failed to list documentation categories')
    }
  }

  /**
   * Check documentation service health
   */
  async checkHealth(): Promise<DocumentationHealth> {
    try {
      const response = await axios.get<DocumentationHealth>(`${this.baseURL}/health`)
      return response.data
    } catch (error) {
      console.error('Error checking documentation health:', error)
      throw new Error('Failed to check documentation service health')
    }
  }

  /**
   * Get documentation by category
   */
  async getDocumentationByCategory(category: string): Promise<DocumentationFile[]> {
    try {
      const allDocs = await this.listDocumentation()
      return allDocs.files.filter(doc => 
        doc.category.toLowerCase() === category.toLowerCase()
      )
    } catch (error) {
      console.error(`Error getting documentation by category ${category}:`, error)
      throw new Error(`Failed to get documentation for category: ${category}`)
    }
  }

  /**
   * Search documentation files by title or description
   */
  async searchDocumentation(query: string): Promise<DocumentationFile[]> {
    try {
      const allDocs = await this.listDocumentation()
      const searchTerm = query.toLowerCase()
      
      return allDocs.files.filter(doc => 
        doc.title.toLowerCase().includes(searchTerm) ||
        doc.description.toLowerCase().includes(searchTerm) ||
        doc.filename.toLowerCase().includes(searchTerm)
      )
    } catch (error) {
      console.error(`Error searching documentation:`, error)
      throw new Error('Failed to search documentation')
    }
  }
}

export const documentationService = new DocumentationService()
