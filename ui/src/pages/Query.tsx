import { useState, useEffect } from 'react'
import { 
  Search, 
  Clock, 
  FileText, 
  TrendingUp,
  RefreshCw
} from 'lucide-react'
import { queryService, QueryResponse, QueryHistory } from '../services/queryService'
import CitationBlock from '../components/CitationBlock'
import EvidenceBlock from '../components/EvidenceBlock'
import toast from 'react-hot-toast'

const Query = () => {
  const [query, setQuery] = useState('')
  const [searchResults, setSearchResults] = useState<QueryResponse | null>(null)
  const [queryHistory, setQueryHistory] = useState<QueryHistory[]>([])
  const [loading, setLoading] = useState(false)
  const [maxResults, setMaxResults] = useState(5)
  const [searchType, setSearchType] = useState<'hybrid' | 'semantic' | 'keyword'>('hybrid')
  const [includeMetadata, setIncludeMetadata] = useState(true)

  useEffect(() => {
    fetchQueryHistory()
  }, [])

  const fetchQueryHistory = async () => {
    try {
      const history = await queryService.getQueryHistory(20)
      setQueryHistory(history)
    } catch (error) {
      console.error('Error fetching query history:', error)
    }
  }

  const handleSearch = async () => {
    if (!query.trim()) {
      toast.error('Please enter a query', { position: 'bottom-left', className: 'lg:ml-64' })
      return
    }

    setLoading(true)
    try {
      const results = await queryService.searchDocuments({
        query: query.trim(),
        max_results: maxResults,
        include_metadata: includeMetadata,
        filters: {
          search_type: searchType
        }
      })
      
      setSearchResults(results)
      
      // Add to query history
      const newQuery: QueryHistory = {
        id: Date.now().toString(),
        query: query.trim(),
        response_time: results.response_time,
        timestamp: new Date().toISOString(),
        results_count: results.retrieved_chunks.length
      }
      setQueryHistory(prev => [newQuery, ...prev.slice(0, 19)])
      
      toast.success(`Found ${results.retrieved_chunks.length} results in ${results.response_time.toFixed(2)}s`)
    } catch (error: any) {
      let errorMessage = 'Search failed'
      
      if (error.response?.status === 404) {
        errorMessage = 'Search endpoint not found - please check backend configuration'
      } else if (error.response?.status >= 500) {
        errorMessage = 'Server error - please try again later'
      } else if (error.code === 'NETWORK_ERROR' || error.message === 'Network Error') {
        errorMessage = 'Cannot connect to server - please check if backend is running'
      } else if (error.response?.data?.detail) {
        errorMessage = error.response.data.detail
      }
      
      toast.error(errorMessage, { position: 'bottom-left', className: 'lg:ml-64' })
      console.error('Error searching documents:', error)
    } finally {
      setLoading(false)
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSearch()
    }
  }

  const formatTimestamp = (timestamp: string) => {
    return new Date(timestamp).toLocaleString()
  }

  const getSearchTypeLabel = (type: string) => {
    switch (type) {
      case 'hybrid':
        return 'Hybrid (Semantic + Keyword)'
      case 'semantic':
        return 'Semantic Search'
      case 'keyword':
        return 'Keyword Search'
      default:
        return type
    }
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-secondary-900">Query Documents</h1>
        <p className="mt-2 text-secondary-600">
          Search and analyze your document collection using advanced RAG techniques
        </p>
      </div>

      {/* Search Interface */}
      <div className="card p-6">
        <div className="space-y-4">
          <div>
            <label htmlFor="query" className="block text-sm font-medium text-secondary-700 mb-2">
              Enter your question
            </label>
            <textarea
              id="query"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="e.g., What is the company policy on remote work?"
              className="input w-full h-24 resize-none"
              disabled={loading}
            />
          </div>

          {/* Search Options */}
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
            <div>
              <label className="block text-sm font-medium text-secondary-700 mb-2">
                Max Results
              </label>
              <select
                value={maxResults}
                onChange={(e) => setMaxResults(Number(e.target.value))}
                className="input"
                disabled={loading}
              >
                <option value={3}>3</option>
                <option value={5}>5</option>
                <option value={10}>10</option>
                <option value={20}>20</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-secondary-700 mb-2">
                Search Type
              </label>
              <select
                value={searchType}
                onChange={(e) => setSearchType(e.target.value as any)}
                className="input"
                disabled={loading}
              >
                <option value="hybrid">Hybrid</option>
                <option value="semantic">Semantic</option>
                <option value="keyword">Keyword</option>
              </select>
            </div>

            <div className="flex items-end">
              <label className="flex items-center">
                <input
                  type="checkbox"
                  checked={includeMetadata}
                  onChange={(e) => setIncludeMetadata(e.target.checked)}
                  className="rounded border-secondary-300 text-primary-600 focus:ring-primary-500"
                  disabled={loading}
                />
                <span className="ml-2 text-sm text-secondary-700">Include Metadata</span>
              </label>
            </div>
          </div>

          <div className="flex items-center space-x-3">
            <button
              onClick={handleSearch}
              disabled={loading || !query.trim()}
              className="btn btn-primary"
            >
              {loading ? (
                <>
                  <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                  Searching...
                </>
              ) : (
                <>
                  <Search className="h-4 w-4 mr-2" />
                  Search
                </>
              )}
            </button>
            
            <button
              onClick={() => {
                setQuery('')
                setSearchResults(null)
              }}
              className="btn btn-outline"
              disabled={loading}
            >
              Clear
            </button>
          </div>
        </div>
      </div>

      {/* Search Results */}
      {searchResults && (
        <div className="space-y-6">
          {/* Results Summary */}
          <div className="card p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-medium text-secondary-900">Search Results</h3>
              <div className="flex items-center space-x-4 text-sm text-secondary-600">
                <span className="flex items-center">
                  <FileText className="h-4 w-4 mr-1" />
                  {searchResults.retrieved_chunks.length} results
                </span>
                <span className="flex items-center">
                  <Clock className="h-4 w-4 mr-1" />
                  {searchResults.response_time.toFixed(2)}s
                </span>
                <span className="flex items-center">
                  <TrendingUp className="h-4 w-4 mr-1" />
                  {getSearchTypeLabel(searchType)}
                </span>
              </div>
            </div>

            {/* Citations and Evidence */}
            {searchResults.citations && searchResults.citations.length > 0 ? (
              <CitationBlock 
                citations={searchResults.citations}
                answer={searchResults.answer}
              />
            ) : searchResults.retrieved_chunks && searchResults.retrieved_chunks.length > 0 ? (
              <EvidenceBlock 
                evidence={searchResults.retrieved_chunks.map(chunk => ({
                  id: chunk.chunk.id,
                  document_id: chunk.document.id,
                  document_title: chunk.document.metadata?.title || chunk.document.id,
                  page_number: chunk.chunk.metadata?.page_number,
                  section_title: chunk.chunk.metadata?.section_title,
                  chunk_id: chunk.chunk.id,
                  score: chunk.score,
                  content: chunk.chunk.content,
                  metadata: chunk.chunk.metadata,
                  timestamp: chunk.document.metadata?.created_at,
                  author: chunk.document.metadata?.author
                }))}
                query={query}
              />
            ) : (
              <>
                {/* Fallback: AI Answer */}
                {searchResults.answer && (
                  <div className="mb-6 p-4 bg-primary-50 border border-primary-200 rounded-lg">
                    <h4 className="font-medium text-primary-900 mb-2">AI Answer</h4>
                    <p className="text-primary-800 whitespace-pre-wrap">{searchResults.answer}</p>
                  </div>
                )}

                {/* Fallback: Retrieved Chunks */}
                <div className="space-y-4">
                  {searchResults.retrieved_chunks.map((chunk, index) => (
                    <div key={chunk.chunk.id} className="border border-secondary-200 rounded-lg p-4">
                      <div className="flex items-center justify-between mb-2">
                        <h5 className="font-medium text-secondary-900">
                          Result {index + 1}
                        </h5>
                        <div className="flex items-center space-x-2">
                          <span className="text-sm text-secondary-600">
                            Score: {chunk.score.toFixed(3)}
                          </span>
                          <span className="text-sm text-secondary-500">
                            Document: {chunk.document.id}
                          </span>
                        </div>
                      </div>
                      
                      <div className="text-sm text-secondary-700 mb-3">
                        {chunk.chunk.content}
                      </div>

                      {includeMetadata && chunk.chunk.metadata && Object.keys(chunk.chunk.metadata).length > 0 && (
                        <details className="text-sm">
                          <summary className="cursor-pointer text-secondary-600 hover:text-secondary-800">
                            View Metadata
                          </summary>
                          <div className="mt-2 p-3 bg-secondary-50 rounded border">
                            <pre className="text-xs text-secondary-700 overflow-x-auto">
                              {JSON.stringify(chunk.chunk.metadata, null, 2)}
                            </pre>
                          </div>
                        </details>
                      )}
                    </div>
                  ))}
                </div>
              </>
            )}
          </div>
        </div>
      )}

      {/* Query History */}
      {queryHistory.length > 0 && (
        <div className="card p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-medium text-secondary-900">Recent Queries</h3>
            <button
              onClick={fetchQueryHistory}
              className="btn btn-outline btn-sm"
            >
              <RefreshCw className="h-4 w-4 mr-2" />
              Refresh
            </button>
          </div>
          
          <div className="space-y-3">
            {queryHistory.slice(0, 10).map((query) => (
              <div key={query.id} className="flex items-center justify-between py-2 border-b border-secondary-100 last:border-b-0">
                <div className="flex-1">
                  <p className="text-sm text-secondary-900 truncate">{query.query}</p>
                  <p className="text-xs text-secondary-500">
                    {formatTimestamp(query.timestamp)}
                  </p>
                </div>
                <div className="flex items-center space-x-4 text-sm text-secondary-600">
                  <span>{query.results_count} results</span>
                  <span>{query.response_time.toFixed(2)}s</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

export default Query
