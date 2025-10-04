import { useState, useEffect } from 'react'
import { 
  Database,
  Clock,
  FileText,
  Search,
  RefreshCw
} from 'lucide-react'
import { adminService, SystemStats, HealthCheck } from '../services/adminService'
import { documentService, ProcessingStatus } from '../services/documentService'
import { queryService, QueryHistory } from '../services/queryService'
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer, 
  PieChart, 
  Pie, 
  Cell,
  BarChart,
  Bar,
  AreaChart,
  Area
} from 'recharts'

const Analytics = () => {
  const [systemStats, setSystemStats] = useState<SystemStats | null>(null)
  const [healthCheck, setHealthCheck] = useState<HealthCheck | null>(null)
  const [processingStatus, setProcessingStatus] = useState<ProcessingStatus | null>(null)
  const [queryHistory, setQueryHistory] = useState<QueryHistory[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetchAnalyticsData()
  }, [])

  const fetchAnalyticsData = async () => {
    try {
      setLoading(true)
      const [stats, health, status, queries] = await Promise.all([
        adminService.getSystemStats(),
        adminService.getHealthCheck(),
        documentService.getProcessingStatus(),
        queryService.getQueryHistory(100)
      ])
      
      setSystemStats(stats)
      setHealthCheck(health)
      setProcessingStatus(status)
      setQueryHistory(queries)
    } catch (error) {
      console.error('Failed to fetch analytics data:', error)
    } finally {
      setLoading(false)
    }
  }

  // Generate sample performance data
  const performanceData = [
    { time: '00:00', responseTime: 450, queries: 12, throughput: 0.2 },
    { time: '02:00', responseTime: 380, queries: 8, throughput: 0.13 },
    { time: '04:00', responseTime: 420, queries: 5, throughput: 0.08 },
    { time: '06:00', responseTime: 350, queries: 3, throughput: 0.05 },
    { time: '08:00', responseTime: 600, queries: 25, throughput: 0.42 },
    { time: '10:00', responseTime: 520, queries: 35, throughput: 0.58 },
    { time: '12:00', responseTime: 480, queries: 28, throughput: 0.47 },
    { time: '14:00', responseTime: 410, queries: 22, throughput: 0.37 },
    { time: '16:00', responseTime: 390, queries: 18, throughput: 0.3 },
    { time: '18:00', responseTime: 440, queries: 15, throughput: 0.25 },
    { time: '20:00', responseTime: 460, queries: 20, throughput: 0.33 },
    { time: '22:00', responseTime: 430, queries: 16, throughput: 0.27 },
  ]

  const queryTypesData = [
    { name: 'Factual', value: 45, color: '#3b82f6' },
    { name: 'Comparison', value: 23, color: '#10b981' },
    { name: 'Analytical', value: 18, color: '#f59e0b' },
    { name: 'Procedural', value: 14, color: '#ef4444' },
  ]

  const documentStatusData = [
    { name: 'Completed', value: processingStatus?.processed_documents || 0, color: '#10b981' },
    { name: 'Processing', value: processingStatus?.processing_documents || 0, color: '#f59e0b' },
    { name: 'Failed', value: processingStatus?.failed_documents || 0, color: '#ef4444' },
  ]

  const systemHealthData = [
    { component: 'Database', status: healthCheck?.components.database === 'healthy' ? 100 : 0 },
    { component: 'Cache', status: healthCheck?.components.cache === 'healthy' ? 100 : 0 },
    { component: 'Embeddings', status: healthCheck?.components.embedding_service === 'healthy' ? 100 : 0 },
  ]

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
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-secondary-900">Analytics & Monitoring</h1>
          <p className="mt-2 text-secondary-600">
            System performance, usage statistics, and health monitoring
          </p>
        </div>
        <button
          onClick={fetchAnalyticsData}
          className="btn btn-outline"
        >
          <RefreshCw className="h-4 w-4 mr-2" />
          Refresh Data
        </button>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-4">
        <div className="card p-6">
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <FileText className="h-8 w-8 text-primary-600" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-secondary-600">Total Documents</p>
              <p className="text-2xl font-semibold text-secondary-900">
                {processingStatus?.total_documents || 0}
              </p>
            </div>
          </div>
        </div>

        <div className="card p-6">
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <Search className="h-8 w-8 text-blue-600" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-secondary-600">Total Queries</p>
              <p className="text-2xl font-semibold text-secondary-900">
                {systemStats?.queries.total || 0}
              </p>
            </div>
          </div>
        </div>

        <div className="card p-6">
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <Clock className="h-8 w-8 text-green-600" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-secondary-600">Avg Response Time</p>
              <p className="text-2xl font-semibold text-secondary-900">
                {systemStats?.queries.avg_response_time ? `${systemStats.queries.avg_response_time.toFixed(1)}s` : '0.0s'}
              </p>
            </div>
          </div>
        </div>

        <div className="card p-6">
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <Database className="h-8 w-8 text-purple-600" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-secondary-600">Cache Hit Rate</p>
              <p className="text-2xl font-semibold text-secondary-900">
                {systemStats?.cache.hit_rate ? `${(systemStats.cache.hit_rate * 100).toFixed(1)}%` : '0%'}
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* System Health */}
      <div className="card p-6">
        <h3 className="text-lg font-medium text-secondary-900 mb-4">System Health</h3>
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
          <div>
            <h4 className="text-sm font-medium text-secondary-700 mb-3">Component Status</h4>
            <div className="space-y-3">
              {Object.entries(healthCheck?.components || {}).map(([component, status]) => (
                <div key={component} className="flex items-center justify-between">
                  <span className="text-sm text-secondary-600 capitalize">{component}</span>
                  <div className="flex items-center space-x-2">
                    <div className={`h-2 w-2 rounded-full ${
                      status === 'healthy' ? 'bg-green-500' : 'bg-red-500'
                    }`}></div>
                    <span className="text-sm font-medium">
                      {status === 'healthy' ? 'Healthy' : 'Unhealthy'}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>
          
          <div>
            <h4 className="text-sm font-medium text-secondary-700 mb-3">Health Overview</h4>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={systemHealthData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="component" />
                <YAxis domain={[0, 100]} />
                <Tooltip />
                <Bar dataKey="status" fill="#3b82f6" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Performance Charts */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <div className="card p-6">
          <h3 className="text-lg font-medium text-secondary-900 mb-4">Response Time Over Time</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={performanceData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="time" />
              <YAxis />
              <Tooltip />
              <Line type="monotone" dataKey="responseTime" stroke="#3b82f6" strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="card p-6">
          <h3 className="text-lg font-medium text-secondary-900 mb-4">Query Throughput</h3>
          <ResponsiveContainer width="100%" height={300}>
            <AreaChart data={performanceData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="time" />
              <YAxis />
              <Tooltip />
              <Area type="monotone" dataKey="throughput" stroke="#10b981" fill="#10b981" fillOpacity={0.3} />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Distribution Charts */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <div className="card p-6">
          <h3 className="text-lg font-medium text-secondary-900 mb-4">Query Types Distribution</h3>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={queryTypesData}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                outerRadius={80}
                fill="#8884d8"
                dataKey="value"
              >
                {queryTypesData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </div>

        <div className="card p-6">
          <h3 className="text-lg font-medium text-secondary-900 mb-4">Document Processing Status</h3>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={documentStatusData}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                outerRadius={80}
                fill="#8884d8"
                dataKey="value"
              >
                {documentStatusData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Recent Activity */}
      <div className="card p-6">
        <h3 className="text-lg font-medium text-secondary-900 mb-4">Recent Query Activity</h3>
        {queryHistory.length > 0 ? (
          <div className="space-y-3">
            {queryHistory.slice(0, 10).map((query) => (
              <div key={query.id} className="flex items-center justify-between py-2 border-b border-secondary-100 last:border-b-0">
                <div className="flex-1">
                  <p className="text-sm text-secondary-900 truncate">{query.query}</p>
                  <p className="text-xs text-secondary-500">
                    {new Date(query.timestamp).toLocaleString()}
                  </p>
                </div>
                <div className="flex items-center space-x-4 text-sm text-secondary-600">
                  <span>{query.results_count} results</span>
                  <span>{query.response_time.toFixed(2)}s</span>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-sm text-secondary-500">No recent queries</p>
        )}
      </div>

      {/* System Information */}
      <div className="card p-6">
        <h3 className="text-lg font-medium text-secondary-900 mb-4">System Information</h3>
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
          <div>
            <h4 className="text-sm font-medium text-secondary-700 mb-2">Version</h4>
            <p className="text-sm text-secondary-600">{healthCheck?.version || 'Unknown'}</p>
          </div>
          <div>
            <h4 className="text-sm font-medium text-secondary-700 mb-2">Status</h4>
            <p className="text-sm text-secondary-600 capitalize">{healthCheck?.status || 'Unknown'}</p>
          </div>
          <div>
            <h4 className="text-sm font-medium text-secondary-700 mb-2">Last Updated</h4>
            <p className="text-sm text-secondary-600">
              {healthCheck?.timestamp ? new Date(healthCheck.timestamp).toLocaleString() : 'Unknown'}
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}

export default Analytics
