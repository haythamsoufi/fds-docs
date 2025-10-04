import { useState, useEffect } from 'react'
import { 
  Database, 
  Server, 
  RefreshCw,
  Save,
  AlertTriangle,
  CheckCircle,
  Clock,
  Zap
} from 'lucide-react'
import { adminService, SystemStats, HealthCheck } from '../services/adminService'
import { documentService } from '../services/documentService'
import SystemStatus from '../components/SystemStatus'
import { ModelSelector } from '../components/ModelSelector'
import toast from 'react-hot-toast'

const Settings = () => {
  const [systemStats, setSystemStats] = useState<SystemStats | null>(null)
  const [healthCheck, setHealthCheck] = useState<HealthCheck | null>(null)
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)

  // Configuration state
  const [config, setConfig] = useState({
    apiHost: 'localhost',
    apiPort: 8000,
    debugMode: false,
    maxFileSize: 10485760, // 10MB
    embeddingModel: 'all-MiniLM-L6-v2',
    chunkSize: 1000,
    chunkOverlap: 200,
    retrievalK: 5,
    similarityThreshold: 0.7,
    cacheTtl: 3600,
  })

  useEffect(() => {
    fetchSystemData()
  }, [])

  const fetchSystemData = async () => {
    try {
      setLoading(true)
      const [stats, health] = await Promise.all([
        adminService.getSystemStats(),
        adminService.getHealthCheck()
      ])
      
      setSystemStats(stats)
      setHealthCheck(health)
    } catch (error) {
      console.error('Failed to fetch system data:', error)
    } finally {
      setLoading(false)
    }
  }

  const handleSaveConfig = async () => {
    setSaving(true)
    try {
      // In a real implementation, this would save to the backend
      await new Promise(resolve => setTimeout(resolve, 1000)) // Simulate API call
      toast.success('Configuration saved successfully')
    } catch (error) {
      toast.error('Failed to save configuration', { position: 'bottom-left', className: 'lg:ml-64' })
    } finally {
      setSaving(false)
    }
  }

  const handleClearCache = async () => {
    try {
      await adminService.clearCache()
      toast.success('Cache cleared successfully')
      fetchSystemData()
    } catch (error) {
      toast.error('Failed to clear cache', { position: 'bottom-left', className: 'lg:ml-64' })
    }
  }

  const handleReprocessDocuments = async () => {
    try {
      await documentService.reprocessDocuments()
      toast.success('Document reprocessing started')
      fetchSystemData()
    } catch (error) {
      toast.error('Failed to start reprocessing', { position: 'bottom-left', className: 'lg:ml-64' })
    }
  }

  const handlePopulateVectors = async () => {
    try {
      await adminService.populateVectors()
      toast.success('Vector store population started')
      fetchSystemData()
    } catch (error) {
      toast.error('Failed to start vector population', { position: 'bottom-left', className: 'lg:ml-64' })
    }
  }

  const getHealthStatus = (component: string) => {
    const status = healthCheck?.components[component]
    if (status === 'healthy') {
      return { icon: CheckCircle, color: 'text-green-500', label: 'Healthy' }
    } else if (status === 'unhealthy') {
      return { icon: AlertTriangle, color: 'text-red-500', label: 'Unhealthy' }
    } else {
      return { icon: Clock, color: 'text-yellow-500', label: 'Unknown' }
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600"></div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-secondary-900">System Settings</h1>
        <p className="mt-2 text-secondary-600">
          Configure and manage your Enterprise RAG System
        </p>
      </div>

      {/* Model Selection */}
      <ModelSelector 
        onModelChange={(model) => {
          console.log('Model changed to:', model);
          // Optionally refresh system data after model change
          fetchSystemData();
        }}
        className="mb-6"
      />

      {/* System Status */}
      <div className="card p-6">
        <h3 className="text-lg font-medium text-secondary-900 mb-4">System Status</h3>
        <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-3">
          <div>
            <h4 className="text-sm font-medium text-secondary-700 mb-3">Component Health</h4>
            <div className="space-y-2">
              {Object.entries(healthCheck?.components || {}).map(([component]) => {
                const health = getHealthStatus(component)
                const Icon = health.icon
                return (
                  <div key={component} className="flex items-center justify-between">
                    <span className="text-sm text-secondary-600 capitalize">{component}</span>
                    <div className="flex items-center space-x-2">
                      <Icon className={`h-4 w-4 ${health.color}`} />
                      <span className="text-sm font-medium">{health.label}</span>
                    </div>
                  </div>
                )
              })}
            </div>
          </div>

          <div>
            <h4 className="text-sm font-medium text-secondary-700 mb-3">System Metrics</h4>
            <div className="space-y-2">
              <div className="flex justify-between">
                <span className="text-sm text-secondary-600">Total Documents</span>
                <span className="text-sm font-medium">{systemStats?.documents.total || 0}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm text-secondary-600">Processed</span>
                <span className="text-sm font-medium text-green-600">{systemStats?.documents.processed || 0}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm text-secondary-600">Failed</span>
                <span className="text-sm font-medium text-red-600">{systemStats?.documents.failed || 0}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm text-secondary-600">Total Chunks</span>
                <span className="text-sm font-medium">{systemStats?.chunks.total || 0}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm text-secondary-600">With Embeddings</span>
                <span className="text-sm font-medium text-blue-600">{systemStats?.chunks.with_embeddings || 0}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm text-secondary-600">Embedding Coverage</span>
                <span className="text-sm font-medium">
                  {systemStats?.chunks.embedding_coverage ? `${systemStats.chunks.embedding_coverage.toFixed(1)}%` : '0%'}
                </span>
              </div>
            </div>
          </div>

          <div>
            <h4 className="text-sm font-medium text-secondary-700 mb-3">System Info</h4>
            <div className="space-y-2">
              <div className="flex justify-between">
                <span className="text-sm text-secondary-600">Version</span>
                <span className="text-sm font-medium">{healthCheck?.version || 'Unknown'}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm text-secondary-600">Status</span>
                <span className="text-sm font-medium capitalize">{healthCheck?.status || 'Unknown'}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm text-secondary-600">Last Updated</span>
                <span className="text-sm font-medium">
                  {healthCheck?.timestamp ? new Date(healthCheck.timestamp).toLocaleString() : 'Unknown'}
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Configuration */}
      <div className="card p-6">
        <h3 className="text-lg font-medium text-secondary-900 mb-4">System Configuration</h3>
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-secondary-700 mb-2">
                API Host
              </label>
              <input
                type="text"
                value={config.apiHost}
                onChange={(e) => setConfig(prev => ({ ...prev, apiHost: e.target.value }))}
                className="input w-full"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-secondary-700 mb-2">
                API Port
              </label>
              <input
                type="number"
                value={config.apiPort}
                onChange={(e) => setConfig(prev => ({ ...prev, apiPort: Number(e.target.value) }))}
                className="input w-full"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-secondary-700 mb-2">
                Max File Size (bytes)
              </label>
              <input
                type="number"
                value={config.maxFileSize}
                onChange={(e) => setConfig(prev => ({ ...prev, maxFileSize: Number(e.target.value) }))}
                className="input w-full"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-secondary-700 mb-2">
                Embedding Model
              </label>
              <select
                value={config.embeddingModel}
                onChange={(e) => setConfig(prev => ({ ...prev, embeddingModel: e.target.value }))}
                className="input w-full"
              >
                <option value="all-MiniLM-L6-v2">all-MiniLM-L6-v2</option>
                <option value="all-mpnet-base-v2">all-mpnet-base-v2</option>
                <option value="embaas/sentence-transformers-e5-large-v2">e5-large-v2</option>
              </select>
            </div>
          </div>

          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-secondary-700 mb-2">
                Chunk Size
              </label>
              <input
                type="number"
                value={config.chunkSize}
                onChange={(e) => setConfig(prev => ({ ...prev, chunkSize: Number(e.target.value) }))}
                className="input w-full"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-secondary-700 mb-2">
                Chunk Overlap
              </label>
              <input
                type="number"
                value={config.chunkOverlap}
                onChange={(e) => setConfig(prev => ({ ...prev, chunkOverlap: Number(e.target.value) }))}
                className="input w-full"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-secondary-700 mb-2">
                Retrieval K
              </label>
              <input
                type="number"
                value={config.retrievalK}
                onChange={(e) => setConfig(prev => ({ ...prev, retrievalK: Number(e.target.value) }))}
                className="input w-full"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-secondary-700 mb-2">
                Similarity Threshold
              </label>
              <input
                type="number"
                step="0.1"
                min="0"
                max="1"
                value={config.similarityThreshold}
                onChange={(e) => setConfig(prev => ({ ...prev, similarityThreshold: Number(e.target.value) }))}
                className="input w-full"
              />
            </div>
          </div>
        </div>

        <div className="mt-6 flex items-center justify-between">
          <div className="flex items-center">
            <input
              type="checkbox"
              checked={config.debugMode}
              onChange={(e) => setConfig(prev => ({ ...prev, debugMode: e.target.checked }))}
              className="rounded border-secondary-300 text-primary-600 focus:ring-primary-500"
            />
            <label className="ml-2 text-sm text-secondary-700">Debug Mode</label>
          </div>
          
          <button
            onClick={handleSaveConfig}
            disabled={saving}
            className="btn btn-primary"
          >
            {saving ? (
              <>
                <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                Saving...
              </>
            ) : (
              <>
                <Save className="h-4 w-4 mr-2" />
                Save Configuration
              </>
            )}
          </button>
        </div>
      </div>

      {/* System Actions */}
      <div className="card p-6">
        <h3 className="text-lg font-medium text-secondary-900 mb-4">System Actions</h3>
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <button
            onClick={fetchSystemData}
            className="btn btn-outline"
          >
            <RefreshCw className="h-4 w-4 mr-2" />
            Refresh Status
          </button>

          <button
            onClick={handleClearCache}
            className="btn btn-secondary"
          >
            <Database className="h-4 w-4 mr-2" />
            Clear Cache
          </button>

          <button
            onClick={handleReprocessDocuments}
            className="btn btn-secondary"
          >
            <Server className="h-4 w-4 mr-2" />
            Reprocess Documents
          </button>

          <button
            onClick={handlePopulateVectors}
            className="btn btn-primary"
          >
            <Zap className="h-4 w-4 mr-2" />
            Populate Vectors
          </button>
        </div>
      </div>

      {/* Environment Information */}
      <div className="card p-6">
        <h3 className="text-lg font-medium text-secondary-900 mb-4">Environment Information</h3>
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
          <div>
            <h4 className="text-sm font-medium text-secondary-700 mb-2">Runtime</h4>
            <div className="space-y-1">
              <div className="flex justify-between">
                <span className="text-sm text-secondary-600">Python Version</span>
                <span className="text-sm font-medium">3.11.0</span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm text-secondary-600">FastAPI Version</span>
                <span className="text-sm font-medium">0.104.1</span>
              </div>
            </div>
          </div>

          <div>
            <h4 className="text-sm font-medium text-secondary-700 mb-2">System Status</h4>
            <SystemStatus className="border-0 p-0" />
          </div>
        </div>
      </div>
    </div>
  )
}

export default Settings
