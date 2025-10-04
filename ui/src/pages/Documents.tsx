import { useState, useEffect, useRef } from 'react'
import { 
  Upload, 
  FileText, 
  Trash2, 
  RefreshCw, 
  AlertCircle,
  CheckCircle,
  Clock,
  X
} from 'lucide-react'
import { documentService, Document, ProcessingStatus } from '../services/documentService'
import toast from 'react-hot-toast'

const Documents = () => {
  const [documents, setDocuments] = useState<Document[]>([])
  const [processingStatus, setProcessingStatus] = useState<ProcessingStatus | null>(null)
  const [loading, setLoading] = useState(true)
  const [uploading, setUploading] = useState(false)
  const [selectedFiles, setSelectedFiles] = useState<FileList | null>(null)
  const [isDragOver, setIsDragOver] = useState(false)
  const [newlyUploadedFiles, setNewlyUploadedFiles] = useState<Set<string>>(new Set())
  const fileInputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    fetchDocuments(true) // Initial load
    fetchProcessingStatus()
    
    // Set up polling for processing status
    const interval = setInterval(() => {
      fetchDocuments(false) // Not initial load
      fetchProcessingStatus()
    }, 3000) // Poll every 3 seconds
    
    return () => clearInterval(interval)
  }, [])

  const fetchDocuments = async (isInitialLoad = false) => {
    try {
      // Only show loading spinner on initial load or when no documents exist
      if (isInitialLoad || documents.length === 0) {
        setLoading(true)
      }
      const docs = await documentService.getDocuments()
      setDocuments(docs)
    } catch (error) {
      toast.error('Failed to fetch documents', { position: 'bottom-left', className: 'lg:ml-64' })
      console.error('Error fetching documents:', error)
    } finally {
      setLoading(false)
    }
  }

  const fetchProcessingStatus = async () => {
    try {
      const status = await documentService.getProcessingStatus()
      setProcessingStatus(status)
    } catch (error) {
      console.error('Error fetching processing status:', error)
    }
  }

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files
    if (files) {
      setSelectedFiles(files)
    }
  }

  const handleDragOver = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault()
    event.stopPropagation()
    setIsDragOver(true)
  }

  const handleDragEnter = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault()
    event.stopPropagation()
    setIsDragOver(true)
  }

  const handleDragLeave = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault()
    event.stopPropagation()
    setIsDragOver(false)
  }

  const handleDrop = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault()
    event.stopPropagation()
    setIsDragOver(false)

    const files = event.dataTransfer.files
    if (files && files.length > 0) {
      // Validate file types
      const validFiles = Array.from(files).filter(file => {
        const extension = '.' + file.name.split('.').pop()?.toLowerCase()
        return ['.pdf', '.docx', '.txt'].includes(extension)
      })

      if (validFiles.length === 0) {
        toast.error('No valid files found. Please upload PDF, DOCX, or TXT files only.', { 
          position: 'bottom-left', 
          className: 'lg:ml-64' 
        })
        return
      }

      if (validFiles.length < files.length) {
        toast(`Some files were skipped. Only ${validFiles.length} of ${files.length} files are supported.`, {
          position: 'bottom-left',
          className: 'lg:ml-64',
          icon: '⚠️',
          duration: 4000
        })
      }

      // Create a new FileList-like object
      const dataTransfer = new DataTransfer()
      validFiles.forEach(file => dataTransfer.items.add(file))
      setSelectedFiles(dataTransfer.files)
    }
  }

  const handleUpload = async () => {
    if (!selectedFiles || selectedFiles.length === 0) {
      toast.error('Please select files to upload', { position: 'bottom-left', className: 'lg:ml-64' })
      return
    }

    setUploading(true)
    
    // Track newly uploaded files for processing indicators
    const uploadedFileNames = Array.from(selectedFiles).map(file => file.name)
    setNewlyUploadedFiles(new Set(uploadedFileNames))
    
    const uploadPromises = Array.from(selectedFiles).map(file => 
      documentService.uploadDocument(file)
    )

    try {
      await Promise.all(uploadPromises)
      toast.success(`Successfully uploaded ${selectedFiles.length} file(s). Processing started...`)
      setSelectedFiles(null)
      if (fileInputRef.current) {
        fileInputRef.current.value = ''
      }
      // Refresh once immediately - the existing polling will handle updates
      fetchDocuments(false)
      fetchProcessingStatus()
      
      // Clear newly uploaded files after a delay to allow processing to complete
      setTimeout(() => {
        setNewlyUploadedFiles(new Set())
      }, 30000) // 30 seconds should be enough for most documents to process
      
    } catch (error) {
      toast.error('Failed to upload files', { position: 'bottom-left', className: 'lg:ml-64' })
      console.error('Error uploading files:', error)
      // Clear newly uploaded files on error
      setNewlyUploadedFiles(new Set())
    } finally {
      setUploading(false)
    }
  }

  const handleDeleteDocument = async (id: string) => {
    if (!confirm('Are you sure you want to delete this document?')) {
      return
    }

    try {
      await documentService.deleteDocument(id)
      toast.success('Document deleted successfully')
      fetchDocuments(false)
      fetchProcessingStatus()
    } catch (error) {
      toast.error('Failed to delete document', { position: 'bottom-left', className: 'lg:ml-64' })
      console.error('Error deleting document:', error)
    }
  }

  const handleReprocessDocuments = async () => {
    try {
      await documentService.reprocessDocuments()
      toast.success('Document reprocessing started')
      fetchDocuments(false)
      fetchProcessingStatus()
    } catch (error) {
      toast.error('Failed to start reprocessing', { position: 'bottom-left', className: 'lg:ml-64' })
      console.error('Error reprocessing documents:', error)
    }
  }

  const handleCleanupOrphanedFiles = async () => {
    if (!confirm('This will delete orphaned files and chunks that are no longer referenced. Continue?')) {
      return
    }

    try {
      const response = await fetch('/api/v1/documents/cleanup', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      })

      if (!response.ok) {
        throw new Error('Cleanup failed')
      }

      const result = await response.json()
      const stats = result.stats
      
      toast.success(
        `Cleanup completed: ${stats.orphaned_files_deleted} files, ${stats.orphaned_chunks_deleted} chunks, ${stats.vector_chunks_cleaned} vectors deleted`,
        { duration: 5000, position: 'bottom-left', className: 'lg:ml-64' }
      )
      
      fetchDocuments(false)
      fetchProcessingStatus()
    } catch (error) {
      toast.error('Failed to cleanup orphaned files', { position: 'bottom-left', className: 'lg:ml-64' })
      console.error('Error cleaning up orphaned files:', error)
    }
  }

  const getStatusIcon = (status: string, filename: string) => {
    // If it's a newly uploaded file and status is not completed, show processing
    const isNewlyUploaded = newlyUploadedFiles.has(filename)
    const isProcessing = status === 'processing' || (isNewlyUploaded && status !== 'completed')
    
    if (isProcessing) {
      return <RefreshCw className="h-5 w-5 text-blue-500 animate-spin" />
    }
    
    switch (status) {
      case 'completed':
        return <CheckCircle className="h-5 w-5 text-green-500" />
      case 'processing':
        return <Clock className="h-5 w-5 text-yellow-500" />
      case 'failed':
        return <AlertCircle className="h-5 w-5 text-red-500" />
      default:
        return <Clock className="h-5 w-5 text-gray-500" />
    }
  }

  const getStatusColor = (status: string, filename: string) => {
    // If it's a newly uploaded file and status is not completed, show processing
    const isNewlyUploaded = newlyUploadedFiles.has(filename)
    const isProcessing = status === 'processing' || (isNewlyUploaded && status !== 'completed')
    
    if (isProcessing) {
      return 'bg-blue-100 text-blue-800'
    }
    
    switch (status) {
      case 'completed':
        return 'bg-green-100 text-green-800'
      case 'processing':
        return 'bg-yellow-100 text-yellow-800'
      case 'failed':
        return 'bg-red-100 text-red-800'
      default:
        return 'bg-gray-100 text-gray-800'
    }
  }

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes'
    const k = 1024
    const sizes = ['Bytes', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-secondary-900">Documents</h1>
          <p className="mt-2 text-secondary-600">
            Manage your document library and processing status
          </p>
        </div>
        <div className="flex items-center space-x-3">
          <button
            onClick={() => fetchDocuments(false)}
            className="btn btn-outline"
          >
            <RefreshCw className="h-4 w-4 mr-2" />
            Refresh
          </button>
          <button
            onClick={handleReprocessDocuments}
            className="btn btn-secondary"
          >
            <RefreshCw className="h-4 w-4 mr-2" />
            Reprocess All
          </button>
          <button
            onClick={handleCleanupOrphanedFiles}
            className="btn btn-warning"
            title="Delete orphaned files and chunks that are no longer referenced"
          >
            <Trash2 className="h-4 w-4 mr-2" />
            Cleanup
          </button>
        </div>
      </div>

      {/* Processing Status */}
      {processingStatus && (
        <div className="card p-6">
          <h3 className="text-lg font-medium text-secondary-900 mb-4">Processing Status</h3>
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-4">
            <div className="text-center">
              <div className="text-2xl font-bold text-secondary-900">
                {processingStatus.total_documents}
              </div>
              <div className="text-sm text-secondary-600">Total Documents</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-green-600">
                {processingStatus.processed_documents}
              </div>
              <div className="text-sm text-secondary-600">Completed</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-yellow-600">
                {processingStatus.processing_documents}
              </div>
              <div className="text-sm text-secondary-600">Processing</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-red-600">
                {processingStatus.failed_documents}
              </div>
              <div className="text-sm text-secondary-600">Failed</div>
            </div>
          </div>
        </div>
      )}

      {/* Upload Section */}
      <div className="card p-6">
        <h3 className="text-lg font-medium text-secondary-900 mb-4">Upload Documents</h3>
        <div className="space-y-4">
          <div 
            className={`border-2 border-dashed rounded-lg p-6 transition-colors duration-200 ${
              isDragOver 
                ? 'border-primary-400 bg-primary-50' 
                : 'border-secondary-300 hover:border-secondary-400'
            }`}
            onDragOver={handleDragOver}
            onDragEnter={handleDragEnter}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
          >
            <div className="text-center">
              <Upload className={`mx-auto h-12 w-12 transition-colors duration-200 ${
                isDragOver ? 'text-primary-500' : 'text-secondary-400'
              }`} />
              <div className="mt-4">
                <label htmlFor="file-upload" className="cursor-pointer">
                  <span className="btn btn-primary">
                    <Upload className="h-4 w-4 mr-2" />
                    Choose Files
                  </span>
                  <input
                    ref={fileInputRef}
                    id="file-upload"
                    type="file"
                    multiple
                    accept=".pdf,.docx,.txt"
                    onChange={handleFileSelect}
                    className="hidden"
                  />
                </label>
                <p className="mt-2 text-sm text-secondary-600">
                  {isDragOver ? 'Drop files here to upload' : 'PDF, DOCX, and TXT files are supported'}
                </p>
                <p className="mt-1 text-xs text-secondary-500">
                  Or drag and drop files here
                </p>
              </div>
            </div>
          </div>

          {selectedFiles && selectedFiles.length > 0 && (
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium text-secondary-900">
                  Selected Files ({selectedFiles.length})
                </span>
                <button
                  onClick={() => {
                    setSelectedFiles(null)
                    if (fileInputRef.current) {
                      fileInputRef.current.value = ''
                    }
                  }}
                  className="text-secondary-400 hover:text-secondary-600"
                >
                  <X className="h-4 w-4" />
                </button>
              </div>
              <div className="space-y-1">
                {Array.from(selectedFiles).map((file, index) => (
                  <div key={index} className="flex items-center justify-between text-sm text-secondary-600">
                    <span>{file.name}</span>
                    <span>{formatFileSize(file.size)}</span>
                  </div>
                ))}
              </div>
              <button
                onClick={handleUpload}
                disabled={uploading}
                className="btn btn-primary w-full"
              >
                {uploading ? (
                  <>
                    <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                    Uploading...
                  </>
                ) : (
                  <>
                    <Upload className="h-4 w-4 mr-2" />
                    Upload Files
                  </>
                )}
              </button>
            </div>
          )}
        </div>
      </div>

      {/* Documents List */}
      <div className="card">
        <div className="px-6 py-4 border-b border-secondary-200">
          <h3 className="text-lg font-medium text-secondary-900">Document Library</h3>
        </div>
        <div className="overflow-x-auto">
          {loading && documents.length === 0 ? (
            <div className="flex items-center justify-center py-12">
              <RefreshCw className="h-8 w-8 animate-spin text-primary-600" />
            </div>
          ) : documents.length > 0 ? (
            <table className="min-w-full divide-y divide-secondary-200">
              <thead className="bg-secondary-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-secondary-500 uppercase tracking-wider">
                    Document
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-secondary-500 uppercase tracking-wider">
                    Status
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-secondary-500 uppercase tracking-wider">
                    Size
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-secondary-500 uppercase tracking-wider">
                    Chunks
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-secondary-500 uppercase tracking-wider">
                    Created
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-secondary-500 uppercase tracking-wider">
                    Actions
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-secondary-200">
                {documents.map((doc) => (
                  <tr key={doc.id} className="hover:bg-secondary-50">
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="flex items-center">
                        <FileText className="h-5 w-5 text-secondary-400 mr-3" />
                        <div>
                          <div className="text-sm font-medium text-secondary-900">
                            {doc.filename}
                          </div>
                          <div className="text-sm text-secondary-500">
                            {doc.id}
                          </div>
                        </div>
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="flex items-center">
                        {getStatusIcon(doc.status, doc.filename)}
                        <span className={`ml-2 inline-flex px-2 py-1 text-xs font-semibold rounded-full ${getStatusColor(doc.status, doc.filename)}`}>
                          {newlyUploadedFiles.has(doc.filename) && doc.status !== 'completed' ? 'processing' : doc.status}
                        </span>
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-secondary-500">
                      {formatFileSize(doc.file_size)}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-secondary-500">
                      {doc.chunk_count}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-secondary-500">
                      {new Date(doc.created_at).toLocaleDateString()}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium">
                      <button
                        onClick={() => handleDeleteDocument(doc.id)}
                        className="text-red-600 hover:text-red-900"
                      >
                        <Trash2 className="h-4 w-4" />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <div className="text-center py-12">
              <FileText className="mx-auto h-12 w-12 text-secondary-400" />
              <h3 className="mt-2 text-sm font-medium text-secondary-900">No documents</h3>
              <p className="mt-1 text-sm text-secondary-500">
                Get started by uploading some documents.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default Documents
