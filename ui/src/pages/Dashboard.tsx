import { useState, useEffect } from 'react'
import { 
  FileText, 
  Search, 
  Clock, 
  CheckCircle
} from 'lucide-react'
import { documentService, ProcessingStatus } from '../services/documentService'
import { adminService, SystemStats, HealthCheck } from '../services/adminService'
import { queryService, QueryHistory } from '../services/queryService'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts'

const Dashboard = () => {
  const [processingStatus, setProcessingStatus] = useState<ProcessingStatus | null>(null)
  const [systemStats, setSystemStats] = useState<SystemStats | null>(null)
  const [healthCheck, setHealthCheck] = useState<HealthCheck | null>(null)
  const [recentQueries, setRecentQueries] = useState<QueryHistory[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const fetchDashboardData = async () => {
      try {
        const [status, stats, health, queries] = await Promise.all([
          documentService.getProcessingStatus(),
          adminService.getSystemStats(),
          adminService.getHealthCheck(),
          queryService.getQueryHistory(10)
        ])
        
        setProcessingStatus(status)
        setSystemStats(stats)
        setHealthCheck(health)
        setRecentQueries(queries)
      } catch (error) {
        console.error('Failed to fetch dashboard data:', error)
      } finally {
        setLoading(false)
      }
    }

    fetchDashboardData()
  }, [])

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600"></div>
      </div>
    )
  }

  const performanceData = [
    { time: '00:00', responseTime: 450, queries: 12 },
    { time: '04:00', responseTime: 380, queries: 8 },
    { time: '08:00', responseTime: 420, queries: 25 },
    { time: '12:00', responseTime: 350, queries: 35 },
    { time: '16:00', responseTime: 400, queries: 28 },
    { time: '20:00', responseTime: 380, queries: 20 },
  ]

  const queryTypesData = [
    { name: 'Factual', value: 45, color: '#3b82f6' },
    { name: 'Comparison', value: 23, color: '#10b981' },
    { name: 'Analytical', value: 18, color: '#f59e0b' },
    { name: 'Procedural', value: 14, color: '#ef4444' },
  ]

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-secondary-900">Dashboard</h1>
        <p className="mt-2 text-secondary-600">
          Overview of your Enterprise RAG System
        </p>
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
              <CheckCircle className="h-8 w-8 text-green-600" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-secondary-600">Processed</p>
              <p className="text-2xl font-semibold text-secondary-900">
                {processingStatus?.processed_documents || 0}
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
              <p className="text-sm font-medium text-secondary-600">Queries Today</p>
              <p className="text-2xl font-semibold text-secondary-900">
                {systemStats?.queries.total || 0}
              </p>
            </div>
          </div>
        </div>

        <div className="card p-6">
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <Clock className="h-8 w-8 text-purple-600" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-secondary-600">Avg Response Time</p>
              <p className="text-2xl font-semibold text-secondary-900">
                {systemStats?.queries.avg_response_time ? `${systemStats.queries.avg_response_time.toFixed(1)}s` : '0.0s'}
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* System Status */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        <div className="card p-6">
          <h3 className="text-lg font-medium text-secondary-900 mb-4">System Health</h3>
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-sm text-secondary-600">Database</span>
              <div className="flex items-center space-x-2">
                <div className={`h-2 w-2 rounded-full ${
                  healthCheck?.components.database === 'healthy' ? 'bg-green-500' : 'bg-red-500'
                }`}></div>
                <span className="text-sm font-medium">
                  {healthCheck?.components.database === 'healthy' ? 'Healthy' : 'Unhealthy'}
                </span>
              </div>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm text-secondary-600">Cache</span>
              <div className="flex items-center space-x-2">
                <div className={`h-2 w-2 rounded-full ${
                  healthCheck?.components.cache === 'healthy' ? 'bg-green-500' : 'bg-red-500'
                }`}></div>
                <span className="text-sm font-medium">
                  {healthCheck?.components.cache === 'healthy' ? 'Healthy' : 'Unhealthy'}
                </span>
              </div>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm text-secondary-600">Embeddings</span>
              <div className="flex items-center space-x-2">
                <div className={`h-2 w-2 rounded-full ${
                  healthCheck?.components.embedding_service === 'healthy' ? 'bg-green-500' : 'bg-red-500'
                }`}></div>
                <span className="text-sm font-medium">
                  {healthCheck?.components.embedding_service === 'healthy' ? 'Healthy' : 'Unhealthy'}
                </span>
              </div>
            </div>
          </div>
        </div>

        <div className="card p-6">
          <h3 className="text-lg font-medium text-secondary-900 mb-4">Processing Status</h3>
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-sm text-secondary-600">Completed</span>
              <span className="text-sm font-medium text-green-600">
                {processingStatus?.processed_documents || 0}
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm text-secondary-600">Processing</span>
              <span className="text-sm font-medium text-yellow-600">
                {processingStatus?.processing_documents || 0}
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm text-secondary-600">Failed</span>
              <span className="text-sm font-medium text-red-600">
                {processingStatus?.failed_documents || 0}
              </span>
            </div>
          </div>
        </div>

        <div className="card p-6">
          <h3 className="text-lg font-medium text-secondary-900 mb-4">Cache Performance</h3>
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-sm text-secondary-600">Hit Rate</span>
              <span className="text-sm font-medium">
                {systemStats?.cache.hit_rate ? `${(systemStats.cache.hit_rate * 100).toFixed(1)}%` : '0%'}
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm text-secondary-600">Total Embeddings</span>
              <span className="text-sm font-medium">
                {systemStats?.chunks.with_embeddings || 0}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Charts */}
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
      </div>

      {/* Recent Activity */}
      <div className="card p-6">
        <h3 className="text-lg font-medium text-secondary-900 mb-4">Recent Queries</h3>
        {recentQueries.length > 0 ? (
          <div className="space-y-3">
            {recentQueries.slice(0, 5).map((query, index) => (
              <div key={index} className="flex items-center justify-between py-2 border-b border-secondary-100 last:border-b-0">
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
    </div>
  )
}

export default Dashboard
