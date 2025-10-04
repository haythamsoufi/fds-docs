import { useState, useEffect } from 'react'
import { 
  Search, 
  Clock, 
  FileText, 
  TrendingUp,
  RefreshCw,
  Brain,
  BookOpen,
  Zap,
  Target,
  Rocket,
  Microscope,
  BookMarked,
  Sparkles,
  Search as SearchIcon,
  Puzzle,
  Theater,
  Settings,
  Palette,
  Eye,
  Plane
} from 'lucide-react'
import { queryService, QueryResponse, QueryHistory, LLMStatus } from '../services/queryService'
import CitationBlock from '../components/CitationBlock'
import EvidenceBlock from '../components/EvidenceBlock'
import toast from 'react-hot-toast'

const Query = () => {
  const [query, setQuery] = useState('')
  const [searchResults, setSearchResults] = useState<QueryResponse | null>(null)
  const [queryHistory, setQueryHistory] = useState<QueryHistory[]>([])
  const [loading, setLoading] = useState(false)
  const [loadingMessage, setLoadingMessage] = useState<{ icon: any, text: string } | null>(null)
  const [maxResults, setMaxResults] = useState(10)
  const [searchType, setSearchType] = useState<'hybrid' | 'semantic' | 'keyword'>('hybrid')
  const [includeMetadata, setIncludeMetadata] = useState(true)
  const [llmStatus, setLlmStatus] = useState<LLMStatus | null>(null)
  const [showSearchHelp, setShowSearchHelp] = useState(false)
  
  // Suggested questions
  const suggestedQuestions = [
    "In 2024, how many emergency operations were active in Syria?",
    "What are the key priorities for 2025?",
    "Summarize the financial outlook and budget allocation",
    "What emergency response procedures should be followed?",
    "Compare disaster preparedness across different regions",
    "What are the main challenges facing humanitarian aid?",
    "How has climate change affected emergency operations?",
    "What training programs are available for volunteers?",
    "What partnerships exist with local organizations?",
    "What are the success metrics for emergency responses?"
  ]

  // Fun loading messages with SVG icons
  const loadingMessages = [
    { icon: SearchIcon, text: "Digging through documents..." },
    { icon: Brain, text: "Consulting the AI oracle..." },
    { icon: BookOpen, text: "Scanning the knowledge base..." },
    { icon: Zap, text: "Processing your query..." },
    { icon: Search, text: "Hunting for answers..." },
    { icon: Sparkles, text: "Connecting the dots..." },
    { icon: Target, text: "Finding the perfect match..." },
    { icon: Rocket, text: "Launching search sequence..." },
    { icon: Microscope, text: "Analyzing content..." },
    { icon: BookMarked, text: "Reading between the lines..." },
    { icon: Theater, text: "Performing search magic..." },
    { icon: Sparkles, text: "Uncovering hidden gems..." },
    { icon: SearchIcon, text: "Following the paper trail..." },
    { icon: Puzzle, text: "Piecing together clues..." },
    { icon: Theater, text: "Consulting the document oracle..." },
    { icon: Settings, text: "Tuning the search engine..." },
    { icon: Palette, text: "Crafting the perfect answer..." },
    { icon: Eye, text: "Predicting what you need..." },
    { icon: Theater, text: "Putting on the search show..." },
    { icon: Plane, text: "Aerial scanning in progress..." }
  ]

  useEffect(() => {
    fetchQueryHistory()
    fetchLLMStatus()
  }, [])

  const getRandomLoadingMessage = () => {
    return loadingMessages[Math.floor(Math.random() * loadingMessages.length)]
  }

  const fetchQueryHistory = async () => {
    try {
      const history = await queryService.getQueryHistory(20)
      setQueryHistory(history)
    } catch (error) {
      console.error('Error fetching query history:', error)
    }
  }

  const fetchLLMStatus = async () => {
    try {
      const status = await queryService.getLLMStatus()
      setLlmStatus(status)
    } catch (error) {
      console.error('Error fetching LLM status:', error)
    }
  }

  const handleSearch = async () => {
    if (!query.trim()) {
      toast.error('Please enter a query', { position: 'bottom-left', className: 'lg:ml-64' })
      return
    }

    setLoading(true)
    setLoadingMessage(getRandomLoadingMessage())
    
    // Set up interval to cycle through loading messages
    const messageInterval = setInterval(() => {
      setLoadingMessage(getRandomLoadingMessage())
    }, 4000) // Change message every 4 seconds
    
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
      clearInterval(messageInterval)
      setLoading(false)
      setLoadingMessage(null)
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

      {/* LLM Status Indicator */}
      {llmStatus && (
        <div className="card p-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className={`w-3 h-3 rounded-full ${
                llmStatus.status === 'available' 
                  ? 'bg-green-500' 
                  : llmStatus.status === 'configured_but_unavailable'
                  ? 'bg-yellow-500'
                  : 'bg-red-500'
              }`}></div>
              <div>
                <h3 className="font-medium text-secondary-900">
                  AI Response Mode: {
                    llmStatus.response_mode === 'llm_generated' 
                      ? 'LLM Generated' 
                      : 'Extractive Summary'
                  }
                </h3>
                <p className="text-sm text-secondary-600">
                  {llmStatus.status === 'available' 
                    ? `Using ${llmStatus.llm_model || 'AI model'} for enhanced responses`
                    : llmStatus.status === 'configured_but_unavailable'
                    ? 'LLM configured but unavailable - using extractive summary'
                    : 'No LLM configured - using extractive summary for responses'
                  }
                </p>
              </div>
            </div>
            <button
              onClick={fetchLLMStatus}
              className="btn btn-outline btn-sm"
              title="Refresh LLM status"
            >
              <RefreshCw className="h-4 w-4" />
            </button>
          </div>
          
          {llmStatus.status === 'configured_but_unavailable' && (
            <div className="mt-3 p-3 bg-yellow-50 border border-yellow-200 rounded-lg">
              <p className="text-sm text-yellow-800">
                <strong>Note:</strong> Your local LLM server appears to be down. 
                Responses will use extractive summaries instead of AI-generated answers.
                {llmStatus.base_url && (
                  <span className="block mt-1">
                    Expected at: <code className="bg-yellow-100 px-1 rounded">{llmStatus.base_url}</code>
                  </span>
                )}
              </p>
            </div>
          )}
          
          {llmStatus.status === 'not_configured' && (
            <div className="mt-3 p-3 bg-blue-50 border border-blue-200 rounded-lg">
              <p className="text-sm text-blue-800">
                <strong>Info:</strong> No LLM is configured. Responses use extractive summaries 
                from retrieved document chunks. Configure an LLM for enhanced AI-generated responses.
              </p>
            </div>
          )}
        </div>
      )}

      {/* Search Interface */}
      <div className="card p-6">
        <div className="space-y-4">
          {/* Search Help Toggle */}
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-medium text-secondary-900">Search Configuration</h2>
            <button
              onClick={() => setShowSearchHelp(!showSearchHelp)}
              className="flex items-center space-x-2 text-sm text-primary-600 hover:text-primary-700 transition-colors"
            >
              <svg className={`h-4 w-4 transition-transform ${showSearchHelp ? 'rotate-180' : ''}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
              </svg>
              <span>{showSearchHelp ? 'Hide' : 'Show'} Search Help</span>
            </button>
          </div>

          {/* Detailed Search Help */}
          {showSearchHelp && (
            <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg">
              <h3 className="text-sm font-semibold text-blue-900 mb-3">Understanding Search Types</h3>
              <div className="space-y-4 text-sm">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="p-3 bg-white rounded border border-blue-100">
                    <div className="flex items-center space-x-2 mb-2">
                      <div className="w-3 h-3 bg-blue-500 rounded-full"></div>
                      <h4 className="font-medium text-gray-900">Hybrid Search</h4>
                    </div>
                    <p className="text-gray-600 text-xs mb-2">
                      The most powerful option that combines both semantic and keyword search with advanced reranking.
                    </p>
                    <div className="text-xs">
                      <div className="font-medium text-gray-700 mb-1">How it works:</div>
                      <ul className="list-disc list-inside space-y-1 text-gray-600">
                        <li>Runs both searches in parallel</li>
                        <li>Uses 70% semantic + 30% keyword weighting</li>
                        <li>Applies reciprocal rank fusion</li>
                        <li>Uses cross-encoder reranking</li>
                        <li>Applies MMR diversification</li>
                      </ul>
                    </div>
                  </div>

                  <div className="p-3 bg-white rounded border border-purple-100">
                    <div className="flex items-center space-x-2 mb-2">
                      <div className="w-3 h-3 bg-purple-500 rounded-full"></div>
                      <h4 className="font-medium text-gray-900">Semantic Search</h4>
                    </div>
                    <p className="text-gray-600 text-xs mb-2">
                      Uses AI embeddings to understand meaning and find conceptually similar content.
                    </p>
                    <div className="text-xs">
                      <div className="font-medium text-gray-700 mb-1">How it works:</div>
                      <ul className="list-disc list-inside space-y-1 text-gray-600">
                        <li>Converts query to vector embeddings</li>
                        <li>Searches vector database for similar chunks</li>
                        <li>Applies similarity threshold filtering</li>
                        <li>Returns results by semantic similarity</li>
                      </ul>
                    </div>
                  </div>

                  <div className="p-3 bg-white rounded border border-orange-100">
                    <div className="flex items-center space-x-2 mb-2">
                      <div className="w-3 h-3 bg-orange-500 rounded-full"></div>
                      <h4 className="font-medium text-gray-900">Keyword Search</h4>
                    </div>
                    <p className="text-gray-600 text-xs mb-2">
                      Traditional text matching using the BM25 algorithm for exact word and phrase matching.
                    </p>
                    <div className="text-xs">
                      <div className="font-medium text-gray-700 mb-1">How it works:</div>
                      <ul className="list-disc list-inside space-y-1 text-gray-600">
                        <li>Tokenizes query into search terms</li>
                        <li>Uses LIKE queries to find candidate chunks</li>
                        <li>Applies BM25 scoring algorithm</li>
                        <li>Ranks by term frequency and document length</li>
                      </ul>
                    </div>
                  </div>
                </div>

                <div className="p-3 bg-yellow-50 border border-yellow-200 rounded">
                  <div className="flex items-start space-x-2">
                    <svg className="h-4 w-4 text-yellow-600 mt-0.5 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                    </svg>
                    <div className="text-xs text-yellow-800">
                      <strong>Performance Note:</strong> Hybrid search may take slightly longer as it runs both search methods, 
                      but provides the most comprehensive and accurate results. For simple queries, semantic search alone 
                      often provides excellent results with faster response times.
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}
          <div>
            <label htmlFor="query" className="block text-sm font-medium text-secondary-700 mb-2">
              Enter your question
            </label>
            {/* Suggested Questions */}
            <div className="mb-3">
              <div className="flex flex-wrap gap-2">
                {suggestedQuestions.map((q) => (
                  <button
                    key={q}
                    type="button"
                    onClick={() => setQuery(q)}
                    className={`px-3 py-1 rounded-full text-xs border transition-colors ${
                      loading
                        ? 'bg-gray-100 text-gray-400 border-gray-200 cursor-not-allowed'
                        : 'bg-white text-secondary-700 border-secondary-200 hover:border-secondary-300 hover:bg-secondary-50'
                    }`}
                    disabled={loading}
                    title="Click to use this question"
                  >
                    {q}
                  </button>
                ))}
              </div>
            </div>
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
                <span className="ml-2 text-xs text-secondary-500 font-normal">
                  (Click for details)
                </span>
              </label>
              <select
                value={searchType}
                onChange={(e) => setSearchType(e.target.value as any)}
                className="input"
                disabled={loading}
                title={`Current: ${getSearchTypeLabel(searchType)}`}
              >
                <option value="hybrid">Hybrid (Recommended)</option>
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
              className={`relative overflow-hidden transition-all duration-300 ${
                loading 
                  ? 'bg-gradient-to-r from-primary-500 to-primary-600 hover:from-primary-600 hover:to-primary-700 shadow-lg transform scale-105' 
                  : 'bg-gradient-to-r from-primary-500 to-primary-600 hover:from-primary-600 hover:to-primary-700 hover:shadow-lg hover:transform hover:scale-105'
              } ${
                !query.trim() || loading 
                  ? 'opacity-75 cursor-not-allowed' 
                  : 'opacity-100 cursor-pointer'
              } px-6 py-3 rounded-lg font-medium text-white flex items-center justify-center min-w-[140px]`}
            >
              {loading ? (
                <>
                  <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                  <div className="flex items-center">
                    {loadingMessage?.icon && <loadingMessage.icon className="h-4 w-4 mr-2" />}
                    <span className="truncate">{loadingMessage?.text || 'Searching...'}</span>
                  </div>
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
              className={`px-6 py-3 rounded-lg font-medium transition-all duration-300 ${
                loading 
                  ? 'bg-gray-100 text-gray-400 cursor-not-allowed' 
                  : 'bg-white text-gray-700 border-2 border-gray-300 hover:border-gray-400 hover:bg-gray-50 hover:shadow-md'
              }`}
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
                {llmStatus && llmStatus.response_mode === 'extractive_summary' && (
                  <span className="flex items-center text-orange-600">
                    <FileText className="h-4 w-4 mr-1" />
                    Extractive Summary
                  </span>
                )}
              </div>
            </div>

            {/* Primary Answer Display */}
            {searchResults.answer && (
              <div className="mb-6 p-6 bg-gradient-to-r from-primary-50 to-blue-50 border border-primary-200 rounded-lg">
                <div className="flex items-center justify-between mb-4">
                  <h4 className="text-lg font-semibold text-primary-900">
                    {llmStatus && llmStatus.response_mode === 'extractive_summary' 
                      ? 'Document Summary' 
                      : 'AI Answer'
                    }
                  </h4>
                  <div className="flex items-center space-x-2">
                    {llmStatus && llmStatus.response_mode === 'llm_generated' && (
                      <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800">
                        AI Generated
                      </span>
                    )}
                    {llmStatus && llmStatus.response_mode === 'extractive_summary' && (
                      <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                        Extractive Summary
                      </span>
                    )}
                  </div>
                </div>
                
                {llmStatus && llmStatus.response_mode === 'extractive_summary' && (
                  <p className="text-sm text-primary-600 mb-4">
                    Summary generated from retrieved document chunks
                  </p>
                )}
                
                <div className="prose prose-sm max-w-none">
                  <p className="text-primary-800 whitespace-pre-wrap leading-relaxed">
                    {searchResults.answer}
                  </p>
                </div>
              </div>
            )}

            {/* Citations and Evidence */}
            {searchResults.citations && searchResults.citations.length > 0 ? (
              <CitationBlock 
                citations={searchResults.citations}
                answer={searchResults.answer}
              />
            ) : searchResults.retrieved_chunks && searchResults.retrieved_chunks.length > 0 ? (
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <h4 className="text-lg font-medium text-secondary-900">Supporting Evidence</h4>
                  <span className="text-sm text-secondary-600">
                    {searchResults.retrieved_chunks.length} source{searchResults.retrieved_chunks.length !== 1 ? 's' : ''}
                  </span>
                </div>
                
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
              </div>
            ) : !searchResults.answer && (
              <div className="text-center py-8">
                <div className="text-secondary-500 mb-2">
                  <FileText className="h-12 w-12 mx-auto mb-4 opacity-50" />
                </div>
                <p className="text-secondary-600">
                  No relevant information found for your query.
                </p>
                <p className="text-sm text-secondary-500 mt-2">
                  Try rephrasing your question or using different keywords.
                </p>
              </div>
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
