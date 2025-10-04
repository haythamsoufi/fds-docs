import { useState, useEffect } from 'react'
import { 
  Activity, 
  CheckCircle, 
  XCircle, 
  AlertTriangle, 
  Database, 
  Server, 
  HardDrive 
} from 'lucide-react'

interface SystemStatusProps {
  className?: string
}

interface StatusInfo {
  status: 'online' | 'offline' | 'error' | 'checking'
  lastCheck?: Date
  details?: string
  errorMessage?: string
}

const SystemStatus = ({ className = '' }: SystemStatusProps) => {
  const [apiStatus, setApiStatus] = useState<StatusInfo>({ status: 'checking' })
  const [dbStatus, setDbStatus] = useState<StatusInfo>({ status: 'checking' })
  const [cacheStatus, setCacheStatus] = useState<StatusInfo>({ status: 'checking' })
  const [isExpanded, setIsExpanded] = useState(true)

  useEffect(() => {
    const checkApiStatus = async () => {
      try {
        const url = (import.meta as any).env.VITE_API_URL || 'http://localhost:8080'
        const response = await fetch(`${url}/health`)
        if (response.ok) {
          setApiStatus({ 
            status: 'online', 
            lastCheck: new Date(),
            details: url
          })
        } else {
          setApiStatus({ 
            status: 'error', 
            lastCheck: new Date(),
            details: `HTTP ${response.status}`,
            errorMessage: `API Server Error: ${response.status} ${response.statusText}. Check if the API server is running on ${url}`
          })
        }
      } catch (error) {
        const errorMsg = error instanceof Error ? error.message : 'Unknown error'
        setApiStatus({ 
          status: 'offline', 
          lastCheck: new Date(),
          details: 'Connection failed',
          errorMessage: `API Connection Failed: ${errorMsg}. Unable to reach API server at ${(import.meta as any).env.VITE_API_URL || 'http://localhost:8080'}`
        })
      }
    }

    const checkDbStatus = async () => {
      try {
        const url = (import.meta as any).env.VITE_API_URL || 'http://localhost:8080'
        const response = await fetch(`${url}/health`)
        if (response.ok) {
          const data = await response.json()
          const dbComponent = data.components?.database
          const isHealthy = typeof dbComponent === 'string' && dbComponent.includes('healthy')
          setDbStatus({ 
            status: isHealthy ? 'online' : 'error', 
            lastCheck: new Date(),
            details: 'SQLite',
            errorMessage: isHealthy ? undefined : `Database Error: ${dbComponent || 'Unknown error'}`
          })
        } else {
          setDbStatus({ 
            status: 'error', 
            lastCheck: new Date(),
            details: 'Database error',
            errorMessage: `Database Error: ${response.status} ${response.statusText}. SQLite database may be unavailable. Check file path and permissions.`
          })
        }
      } catch (error) {
        const errorMsg = error instanceof Error ? error.message : 'Unknown error'
        setDbStatus({ 
          status: 'offline', 
          lastCheck: new Date(),
          details: 'Connection failed',
          errorMessage: `Database Connection Failed: ${errorMsg}. Cannot access SQLite database file. Verify file exists and app has permissions.`
        })
      }
    }

    const checkCacheStatus = async () => {
      try {
        const url = (import.meta as any).env.VITE_API_URL || 'http://localhost:8080'
        const response = await fetch(`${url}/health`)
        if (response.ok) {
          const data = await response.json()
          const cacheComponent = data.components?.cache
          const isHealthy = typeof cacheComponent === 'string' && (cacheComponent.includes('healthy') || cacheComponent.includes('unavailable'))
          setCacheStatus({ 
            status: isHealthy ? 'online' : 'error', 
            lastCheck: new Date(),
            details: 'Redis',
            errorMessage: isHealthy ? undefined : `Cache Error: ${cacheComponent || 'Unknown error'}`
          })
        } else {
          setCacheStatus({ 
            status: 'error', 
            lastCheck: new Date(),
            details: 'Cache error',
            errorMessage: `Cache Error: ${response.status} ${response.statusText}. Redis cache may be down or unreachable. Check Redis server status and configuration.`
          })
        }
      } catch (error) {
        const errorMsg = error instanceof Error ? error.message : 'Unknown error'
        setCacheStatus({ 
          status: 'offline', 
          lastCheck: new Date(),
          details: 'Connection failed',
          errorMessage: `Cache Connection Failed: ${errorMsg}. Cannot connect to Redis cache server. Verify Redis is running and connection parameters are correct.`
        })
      }
    }

    const checkAllStatuses = async () => {
      await Promise.all([
        checkApiStatus(),
        checkDbStatus(),
        checkCacheStatus()
      ])
    }

    checkAllStatuses()
    const interval = setInterval(checkAllStatuses, 30000) // Check every 30 seconds
    
    return () => clearInterval(interval)
  }, [])

  const getStatusIcon = (status: StatusInfo['status']) => {
    switch (status) {
      case 'checking':
        return <Activity className="h-3 w-3 animate-spin text-theme-blue-500" />
      case 'online':
        return <CheckCircle className="h-3 w-3 text-theme-green-500" />
      case 'offline':
        return <XCircle className="h-3 w-3 text-ifrc-red-500" />
      case 'error':
        return <AlertTriangle className="h-3 w-3 text-theme-orange-500" />
      default:
        return <Activity className="h-3 w-3 text-ifrc-navy-500" />
    }
  }

  const getOverallStatus = () => {
    const statuses = [apiStatus.status, dbStatus.status, cacheStatus.status]
    if (statuses.includes('offline')) return 'offline'
    if (statuses.includes('error')) return 'error'
    if (statuses.includes('checking')) return 'checking'
    return 'online'
  }

  const overallStatus = getOverallStatus()

  const getOverallStatusText = () => {
    switch (overallStatus) {
      case 'online':
        return 'System Online'
      case 'offline':
        return 'System Offline'
      case 'error':
        return 'System Error'
      case 'checking':
        return 'Checking...'
      default:
        return 'Unknown'
    }
  }

  const getOverallStatusColor = () => {
    switch (overallStatus) {
      case 'online':
        return 'text-theme-green-600'
      case 'offline':
        return 'text-ifrc-red-600'
      case 'error':
        return 'text-theme-orange-600'
      default:
        return 'text-theme-blue-600'
    }
  }

  return (
    <div className={`p-4 border-t border-ifrc-navy-200 overflow-visible ${className}`}>
      <div 
        className="flex items-center space-x-2 text-sm cursor-pointer hover:bg-ifrc-navy-50 rounded-md p-1 -m-1 transition-colors"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        {getStatusIcon(overallStatus)}
        <span className={getOverallStatusColor()}>{getOverallStatusText()}</span>
        <span className="text-xs text-ifrc-navy-400 ml-auto">
          {isExpanded ? '▼' : '▶'}
        </span>
      </div>
      
      {isExpanded && (
        <div className="mt-3 space-y-3 text-xs overflow-visible">
          <div className="group relative">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <Server className="h-3 w-3 text-ifrc-navy-400" />
                <span className="text-ifrc-navy-600">API</span>
              </div>
              <div className="flex items-center space-x-2">
                {getStatusIcon(apiStatus.status)}
                <span className={apiStatus.status === 'online' ? 'text-theme-green-600' : 'text-ifrc-red-600'}>
                  {apiStatus.details}
                </span>
              </div>
            </div>
            {apiStatus.errorMessage && (
              <div className="absolute top-full left-0 right-0 mt-1 p-2 bg-ifrc-red-50 border border-ifrc-red-200 rounded text-ifrc-red-700 text-xs opacity-0 group-hover:opacity-100 transition-opacity duration-200 z-50">
                {apiStatus.errorMessage}
              </div>
            )}
          </div>
          
          <div className="group relative">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <Database className="h-3 w-3 text-ifrc-navy-400" />
                <span className="text-ifrc-navy-600">Database</span>
              </div>
              <div className="flex items-center space-x-2">
                {getStatusIcon(dbStatus.status)}
                <span className={dbStatus.status === 'online' ? 'text-theme-green-600' : 'text-ifrc-red-600'}>
                  {dbStatus.details}
                </span>
              </div>
            </div>
            {dbStatus.errorMessage && (
              <div className="absolute top-full left-0 right-0 mt-1 p-2 bg-ifrc-red-50 border border-ifrc-red-200 rounded text-ifrc-red-700 text-xs opacity-0 group-hover:opacity-100 transition-opacity duration-200 z-50">
                {dbStatus.errorMessage}
              </div>
            )}
          </div>
          
          <div className="group relative">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <HardDrive className="h-3 w-3 text-ifrc-navy-400" />
                <span className="text-ifrc-navy-600">Cache</span>
              </div>
              <div className="flex items-center space-x-2">
                {getStatusIcon(cacheStatus.status)}
                <span className={cacheStatus.status === 'online' ? 'text-theme-green-600' : cacheStatus.status === 'checking' ? 'text-theme-blue-600' : cacheStatus.status === 'error' ? 'text-theme-orange-600' : 'text-ifrc-navy-500'}>
                  {cacheStatus.details}
                </span>
              </div>
            </div>
            {cacheStatus.errorMessage && (
              <div className="absolute top-full left-0 right-0 mt-1 p-2 bg-ifrc-red-50 border border-ifrc-red-200 rounded text-ifrc-red-700 text-xs opacity-0 group-hover:opacity-100 transition-opacity duration-200 z-50">
                {cacheStatus.errorMessage}
              </div>
            )}
          </div>
          
          {apiStatus.lastCheck && (
            <div className="text-xs text-ifrc-navy-400 pt-1 border-t border-ifrc-navy-100">
              Last check: {apiStatus.lastCheck.toLocaleTimeString()}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export default SystemStatus
