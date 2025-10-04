import { useState, useEffect } from 'react'
import { 
  Activity, 
  AlertTriangle, 
  CheckCircle, 
  Clock, 
  TrendingUp,
  RefreshCw,
  BarChart3,
  Server
} from 'lucide-react'

interface DashboardData {
  timestamp: string
  health: {
    status: string
    avg_response_time?: number
    error_rate?: number
    avg_confidence?: number
    recent_queries?: number
    memory_usage?: number
    cpu_usage?: number
    issues?: string[]
  }
  query_metrics: {
    current: {
      total_queries?: number
      avg_response_time?: number
      avg_confidence?: number
      avg_retrieved_chunks?: number
      avg_answer_length?: number
      error_rates?: Record<string, number>
      cache_hit_rate?: number
      search_type_distribution?: Record<string, any>
    }
    trends: Array<{
      timestamp: string
      query_count: number
      avg_response_time: number
      avg_confidence: number
      error_rate: number
    }>
    alerts: Array<{
      type: string
      severity: string
      message: string
      value: number
      threshold: number
    }>
  }
  system_metrics: {
    current: {
      avg_memory_usage?: number
      max_memory_usage?: number
      avg_cpu_usage?: number
      max_cpu_usage?: number
      avg_active_connections?: number
      max_active_connections?: number
      current_connections?: number
    }
    alerts: Array<{
      type: string
      severity: string
      message: string
      value: number
      threshold: number
    }>
  }
  performance_summary: {
    avg_response_time: number
    error_rate: Record<string, number>
    cache_hit_rate: number
    throughput: number
    avg_confidence: number
  }
}

const Monitoring = () => {
  const [dashboardData, setDashboardData] = useState<DashboardData | null>(null)
  const [loading, setLoading] = useState(true)
  const [autoRefresh, setAutoRefresh] = useState(true)

  useEffect(() => {
    fetchDashboardData()
    
    if (autoRefresh) {
      const interval = setInterval(fetchDashboardData, 30000) // 30 seconds
      return () => clearInterval(interval)
    }
  }, [autoRefresh])

  const fetchDashboardData = async () => {
    try {
      const response = await fetch('/api/v1/metrics/dashboard-data')
      if (response.ok) {
        const data = await response.json()
        setDashboardData(data)
      }
    } catch (error) {
      console.error('Error fetching dashboard data:', error)
    } finally {
      setLoading(false)
    }
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy': return 'text-green-600 bg-green-50'
      case 'degraded': return 'text-yellow-600 bg-yellow-50'
      case 'unhealthy': return 'text-red-600 bg-red-50'
      default: return 'text-gray-600 bg-gray-50'
    }
  }

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical': return 'text-red-600 bg-red-50 border-red-200'
      case 'warning': return 'text-yellow-600 bg-yellow-50 border-yellow-200'
      default: return 'text-gray-600 bg-gray-50 border-gray-200'
    }
  }

  const formatPercentage = (value: number) => `${(value * 100).toFixed(1)}%`
  const formatBytes = (bytes: number) => {
    const sizes = ['Bytes', 'KB', 'MB', 'GB']
    if (bytes === 0) return '0 Bytes'
    const i = Math.floor(Math.log(bytes) / Math.log(1024))
    return Math.round(bytes / Math.pow(1024, i) * 100) / 100 + ' ' + sizes[i]
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <RefreshCw className="h-8 w-8 animate-spin text-primary-600" />
        <span className="ml-2 text-secondary-600">Loading dashboard...</span>
      </div>
    )
  }

  if (!dashboardData) {
    return (
      <div className="text-center py-12">
        <AlertTriangle className="h-12 w-12 text-red-500 mx-auto mb-4" />
        <h3 className="text-lg font-medium text-secondary-900 mb-2">Unable to load dashboard</h3>
        <p className="text-secondary-600">Please check the system status and try again.</p>
      </div>
    )
  }

  const allAlerts = [...(dashboardData.query_metrics?.alerts || []), ...(dashboardData.system_metrics?.alerts || [])]
  const criticalAlerts = allAlerts.filter(alert => alert.severity === 'critical')
  const warningAlerts = allAlerts.filter(alert => alert.severity === 'warning')

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-secondary-900">System Monitoring</h1>
          <p className="mt-2 text-secondary-600">
            Real-time monitoring and performance metrics for the RAG system
          </p>
        </div>
        <div className="flex items-center space-x-4">
          <label className="flex items-center">
            <input
              type="checkbox"
              checked={autoRefresh}
              onChange={(e) => setAutoRefresh(e.target.checked)}
              className="rounded border-secondary-300 text-primary-600 focus:ring-primary-500"
            />
            <span className="ml-2 text-sm text-secondary-700">Auto-refresh</span>
          </label>
          <button
            onClick={fetchDashboardData}
            className="btn btn-outline btn-sm"
          >
            <RefreshCw className="h-4 w-4 mr-2" />
            Refresh
          </button>
        </div>
      </div>

      {/* Health Status */}
      <div className="card p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-medium text-secondary-900 flex items-center">
            <Activity className="h-5 w-5 mr-2" />
            System Health
          </h3>
          <span className={`px-3 py-1 rounded-full text-sm font-medium ${getStatusColor(dashboardData.health.status)}`}>
            {dashboardData.health.status.toUpperCase()}
          </span>
        </div>

        {dashboardData.health.issues && dashboardData.health.issues.length > 0 && (
          <div className="mb-4 p-3 bg-yellow-50 border border-yellow-200 rounded-lg">
            <div className="flex items-center">
              <AlertTriangle className="h-5 w-5 text-yellow-600 mr-2" />
              <span className="text-sm font-medium text-yellow-800">Issues Detected:</span>
            </div>
            <ul className="mt-2 text-sm text-yellow-700">
              {dashboardData.health.issues.map((issue, index) => (
                <li key={index} className="capitalize">• {issue.replace(/_/g, ' ')}</li>
              ))}
            </ul>
          </div>
        )}

        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
            <div className="flex items-center">
              <Clock className="h-5 w-5 text-blue-600 mr-2" />
              <div>
                <p className="text-sm font-medium text-blue-900">Avg Response Time</p>
                <p className="text-2xl font-bold text-blue-600">
                  {dashboardData.health.avg_response_time?.toFixed(2) || '0.00'}s
                </p>
              </div>
            </div>
          </div>

          <div className="bg-green-50 border border-green-200 rounded-lg p-4">
            <div className="flex items-center">
              <CheckCircle className="h-5 w-5 text-green-600 mr-2" />
              <div>
                <p className="text-sm font-medium text-green-900">Confidence Score</p>
                <p className="text-2xl font-bold text-green-600">
                  {formatPercentage(dashboardData.health.avg_confidence || 0)}
                </p>
              </div>
            </div>
          </div>

          <div className="bg-purple-50 border border-purple-200 rounded-lg p-4">
            <div className="flex items-center">
              <TrendingUp className="h-5 w-5 text-purple-600 mr-2" />
              <div>
                <p className="text-sm font-medium text-purple-900">Recent Queries</p>
                <p className="text-2xl font-bold text-purple-600">
                  {dashboardData.health.recent_queries || 0}
                </p>
              </div>
            </div>
          </div>

          <div className="bg-orange-50 border border-orange-200 rounded-lg p-4">
            <div className="flex items-center">
              <AlertTriangle className="h-5 w-5 text-orange-600 mr-2" />
              <div>
                <p className="text-sm font-medium text-orange-900">Error Rate</p>
                <p className="text-2xl font-bold text-orange-600">
                  {formatPercentage(dashboardData.health.error_rate || 0)}
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Alerts */}
      {allAlerts.length > 0 && (
        <div className="card p-6">
          <h3 className="text-lg font-medium text-secondary-900 mb-4 flex items-center">
            <AlertTriangle className="h-5 w-5 mr-2" />
            Active Alerts
          </h3>

          <div className="space-y-3">
            {criticalAlerts.map((alert, index) => (
              <div key={index} className={`p-4 rounded-lg border ${getSeverityColor(alert.severity)}`}>
                <div className="flex items-center justify-between">
                  <div>
                    <p className="font-medium">{alert.message}</p>
                    <p className="text-sm opacity-75">
                      Current: {typeof alert.value === 'number' ? alert.value.toFixed(2) : alert.value} | 
                      Threshold: {typeof alert.threshold === 'number' ? alert.threshold.toFixed(2) : alert.threshold}
                    </p>
                  </div>
                  <span className="px-2 py-1 rounded text-xs font-medium bg-red-100 text-red-800">
                    CRITICAL
                  </span>
                </div>
              </div>
            ))}

            {warningAlerts.map((alert, index) => (
              <div key={index} className={`p-4 rounded-lg border ${getSeverityColor(alert.severity)}`}>
                <div className="flex items-center justify-between">
                  <div>
                    <p className="font-medium">{alert.message}</p>
                    <p className="text-sm opacity-75">
                      Current: {typeof alert.value === 'number' ? alert.value.toFixed(2) : alert.value} | 
                      Threshold: {typeof alert.threshold === 'number' ? alert.threshold.toFixed(2) : alert.threshold}
                    </p>
                  </div>
                  <span className="px-2 py-1 rounded text-xs font-medium bg-yellow-100 text-yellow-800">
                    WARNING
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Performance Metrics */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Query Performance */}
        <div className="card p-6">
          <h3 className="text-lg font-medium text-secondary-900 mb-4 flex items-center">
            <BarChart3 className="h-5 w-5 mr-2" />
            Query Performance
          </h3>

          <div className="space-y-4">
            <div className="flex justify-between items-center">
              <span className="text-sm text-secondary-600">Total Queries</span>
              <span className="font-medium">{dashboardData.query_metrics?.current?.total_queries || 0}</span>
            </div>

            <div className="flex justify-between items-center">
              <span className="text-sm text-secondary-600">Avg Response Time</span>
              <span className="font-medium">
                {dashboardData.query_metrics?.current?.avg_response_time?.toFixed(2) || '0.00'}s
              </span>
            </div>

            <div className="flex justify-between items-center">
              <span className="text-sm text-secondary-600">Cache Hit Rate</span>
              <span className="font-medium">
                {formatPercentage(dashboardData.query_metrics?.current?.cache_hit_rate || 0)}
              </span>
            </div>

            <div className="flex justify-between items-center">
              <span className="text-sm text-secondary-600">Avg Retrieved Chunks</span>
              <span className="font-medium">
                {dashboardData.query_metrics?.current?.avg_retrieved_chunks?.toFixed(1) || '0.0'}
              </span>
            </div>

            <div className="flex justify-between items-center">
              <span className="text-sm text-secondary-600">Avg Answer Length</span>
              <span className="font-medium">
                {dashboardData.query_metrics?.current?.avg_answer_length?.toFixed(0) || '0'} chars
              </span>
            </div>
          </div>
        </div>

        {/* System Resources */}
        <div className="card p-6">
          <h3 className="text-lg font-medium text-secondary-900 mb-4 flex items-center">
            <Server className="h-5 w-5 mr-2" />
            System Resources
          </h3>

          <div className="space-y-4">
            <div className="flex justify-between items-center">
              <span className="text-sm text-secondary-600">Memory Usage</span>
              <span className="font-medium">
                {formatBytes(dashboardData.system_metrics?.current?.avg_memory_usage || 0)}
              </span>
            </div>

            <div className="flex justify-between items-center">
              <span className="text-sm text-secondary-600">CPU Usage</span>
              <span className="font-medium">
                {formatPercentage(dashboardData.system_metrics?.current?.avg_cpu_usage || 0)}
              </span>
            </div>

            <div className="flex justify-between items-center">
              <span className="text-sm text-secondary-600">Active Connections</span>
              <span className="font-medium">
                {dashboardData.system_metrics?.current?.current_connections || 0}
              </span>
            </div>

            <div className="flex justify-between items-center">
              <span className="text-sm text-secondary-600">Max Connections</span>
              <span className="font-medium">
                {dashboardData.system_metrics?.current?.max_active_connections || 0}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Performance Trends */}
      {dashboardData.query_metrics?.trends && dashboardData.query_metrics.trends.length > 0 && (
        <div className="card p-6">
          <h3 className="text-lg font-medium text-secondary-900 mb-4 flex items-center">
            <TrendingUp className="h-5 w-5 mr-2" />
            Performance Trends (Last Hour)
          </h3>

          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-secondary-200">
              <thead className="bg-secondary-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-secondary-500 uppercase tracking-wider">
                    Time
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-secondary-500 uppercase tracking-wider">
                    Queries
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-secondary-500 uppercase tracking-wider">
                    Response Time
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-secondary-500 uppercase tracking-wider">
                    Confidence
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-secondary-500 uppercase tracking-wider">
                    Error Rate
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-secondary-200">
                {dashboardData.query_metrics.trends.slice(-10).map((trend, index) => (
                  <tr key={index}>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-secondary-900">
                      {new Date(trend.timestamp).toLocaleTimeString()}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-secondary-900">
                      {trend.query_count}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-secondary-900">
                      {trend.avg_response_time.toFixed(2)}s
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-secondary-900">
                      {formatPercentage(trend.avg_confidence)}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-secondary-900">
                      {formatPercentage(trend.error_rate)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Last Updated */}
      <div className="text-center text-sm text-secondary-500">
        Last updated: {new Date(dashboardData.timestamp).toLocaleString()}
      </div>
    </div>
  )
}

export default Monitoring