import React, { useState } from 'react'
import { ChevronDown, ChevronUp, FileText, MapPin, Calendar, User } from 'lucide-react'

interface EvidenceItem {
  id: string
  document_id: string
  document_title?: string
  page_number?: number
  section_title?: string
  chunk_id: string
  score: number
  content: string
  metadata?: Record<string, any>
  timestamp?: string
  author?: string
}

interface EvidenceBlockProps {
  evidence: EvidenceItem[]
  query: string
  maxItems?: number
}

const EvidenceBlock: React.FC<EvidenceBlockProps> = ({ 
  evidence, 
  query, 
  maxItems = 5 
}) => {
  const [expandedItems, setExpandedItems] = useState<Set<string>>(new Set())
  const [showAll, setShowAll] = useState(false)

  const toggleItem = (itemId: string) => {
    setExpandedItems(prev => {
      const newSet = new Set(prev)
      if (newSet.has(itemId)) {
        newSet.delete(itemId)
      } else {
        newSet.add(itemId)
      }
      return newSet
    })
  }

  const formatLocation = (item: EvidenceItem) => {
    const parts = []
    if (item.section_title) {
      parts.push(item.section_title)
    }
    if (item.page_number) {
      parts.push(`Page ${item.page_number}`)
    }
    return parts.join(', ') || 'Document'
  }

  const getScoreColor = (score: number) => {
    if (score >= 0.8) return 'text-green-600 bg-green-50 border-green-200'
    if (score >= 0.6) return 'text-yellow-600 bg-yellow-50 border-yellow-200'
    return 'text-red-600 bg-red-50 border-red-200'
  }

  const getScoreLabel = (score: number) => {
    if (score >= 0.8) return 'High Relevance'
    if (score >= 0.6) return 'Medium Relevance'
    return 'Low Relevance'
  }

  const highlightQuery = (text: string, query: string) => {
    if (!query.trim()) return text
    
    const queryWords = query.toLowerCase().split(/\s+/).filter(word => word.length > 2)
    let highlightedText = text
    
    queryWords.forEach(word => {
      const regex = new RegExp(`\\b${word}\\b`, 'gi')
      highlightedText = highlightedText.replace(regex, `<mark class="bg-yellow-200 px-1 rounded">$&</mark>`)
    })
    
    return highlightedText
  }

  const displayedEvidence = showAll ? evidence : evidence.slice(0, maxItems)

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold text-gray-900 flex items-center">
          <FileText className="h-5 w-5 mr-2" />
          Supporting Evidence
        </h3>
        <div className="flex items-center space-x-4 text-sm text-gray-600">
          <span>{evidence.length} sources found</span>
          {evidence.length > maxItems && (
            <button
              onClick={() => setShowAll(!showAll)}
              className="text-blue-600 hover:text-blue-800 font-medium"
            >
              {showAll ? 'Show Less' : `Show All ${evidence.length}`}
            </button>
          )}
        </div>
      </div>

      {/* Evidence Items */}
      <div className="space-y-3">
        {displayedEvidence.map((item, index) => (
          <div key={item.id} className="border border-gray-200 rounded-lg overflow-hidden">
            {/* Header */}
            <div 
              className="p-4 cursor-pointer hover:bg-gray-50 transition-colors"
              onClick={() => toggleItem(item.id)}
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  <div className="flex items-center">
                    {expandedItems.has(item.id) ? (
                      <ChevronUp className="h-4 w-4 text-gray-500" />
                    ) : (
                      <ChevronDown className="h-4 w-4 text-gray-500" />
                    )}
                  </div>
                  <div className="flex-1">
                    <div className="flex items-center space-x-2 mb-1">
                      <span className="text-sm font-medium text-gray-900">
                        Evidence {index + 1}
                      </span>
                      <span className={`px-2 py-1 text-xs font-medium rounded-full border ${getScoreColor(item.score)}`}>
                        {getScoreLabel(item.score)}
                      </span>
                      <span className="text-xs text-gray-500">
                        {item.score.toFixed(3)}
                      </span>
                    </div>
                    <div className="text-sm text-gray-600">
                      {item.document_title || `Document ${item.document_id}`}
                      {formatLocation(item) && (
                        <span className="ml-2">• {formatLocation(item)}</span>
                      )}
                    </div>
                  </div>
                </div>
                <div className="flex items-center space-x-2">
                  {item.author && (
                    <div className="flex items-center text-xs text-gray-500">
                      <User className="h-3 w-3 mr-1" />
                      {item.author}
                    </div>
                  )}
                  {item.timestamp && (
                    <div className="flex items-center text-xs text-gray-500">
                      <Calendar className="h-3 w-3 mr-1" />
                      {new Date(item.timestamp).toLocaleDateString()}
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* Expanded Content */}
            {expandedItems.has(item.id) && (
              <div className="border-t border-gray-200 p-4 bg-gray-50">
                <div className="space-y-4">
                  {/* Content */}
                  <div>
                    <h6 className="text-sm font-medium text-gray-900 mb-2">Content</h6>
                    <div 
                      className="text-sm text-gray-700 leading-relaxed prose prose-sm max-w-none"
                      dangerouslySetInnerHTML={{ 
                        __html: highlightQuery(item.content, query) 
                      }}
                    />
                  </div>

                  {/* Context Information */}
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                      <h6 className="text-sm font-medium text-gray-900 mb-2">Document Info</h6>
                      <div className="bg-white border border-gray-200 rounded p-3">
                        <dl className="space-y-1">
                          <div className="flex">
                            <dt className="text-xs font-medium text-gray-500 w-20">Document:</dt>
                            <dd className="text-xs text-gray-700">{item.document_id}</dd>
                          </div>
                          {item.document_title && (
                            <div className="flex">
                              <dt className="text-xs font-medium text-gray-500 w-20">Title:</dt>
                              <dd className="text-xs text-gray-700">{item.document_title}</dd>
                            </div>
                          )}
                          <div className="flex">
                            <dt className="text-xs font-medium text-gray-500 w-20">Chunk ID:</dt>
                            <dd className="text-xs text-gray-700 font-mono">{item.chunk_id}</dd>
                          </div>
                        </dl>
                      </div>
                    </div>

                    <div>
                      <h6 className="text-sm font-medium text-gray-900 mb-2">Relevance Metrics</h6>
                      <div className="bg-white border border-gray-200 rounded p-3">
                        <dl className="space-y-1">
                          <div className="flex justify-between">
                            <dt className="text-xs font-medium text-gray-500">Similarity Score:</dt>
                            <dd className="text-xs text-gray-700 font-mono">{item.score.toFixed(4)}</dd>
                          </div>
                          <div className="flex justify-between">
                            <dt className="text-xs font-medium text-gray-500">Content Length:</dt>
                            <dd className="text-xs text-gray-700">{item.content.length} chars</dd>
                          </div>
                          {item.section_title && (
                            <div className="flex justify-between">
                              <dt className="text-xs font-medium text-gray-500">Section:</dt>
                              <dd className="text-xs text-gray-700">{item.section_title}</dd>
                            </div>
                          )}
                          {item.page_number && (
                            <div className="flex justify-between">
                              <dt className="text-xs font-medium text-gray-500">Page:</dt>
                              <dd className="text-xs text-gray-700">{item.page_number}</dd>
                            </div>
                          )}
                        </dl>
                      </div>
                    </div>
                  </div>

                  {/* Metadata */}
                  {item.metadata && Object.keys(item.metadata).length > 0 && (
                    <div>
                      <h6 className="text-sm font-medium text-gray-900 mb-2">Additional Metadata</h6>
                      <div className="bg-white border border-gray-200 rounded p-3">
                        <dl className="grid grid-cols-1 gap-2 sm:grid-cols-2">
                          {Object.entries(item.metadata).map(([key, value]) => (
                            <div key={key} className="flex">
                              <dt className="text-xs font-medium text-gray-500 capitalize w-24 flex-shrink-0">
                                {key.replace(/_/g, ' ')}:
                              </dt>
                              <dd className="text-xs text-gray-700 flex-1 break-words">
                                {typeof value === 'object' ? JSON.stringify(value) : String(value)}
                              </dd>
                            </div>
                          ))}
                        </dl>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Summary */}
      {evidence.length > 0 && (
        <div className="mt-4 p-4 bg-blue-50 border border-blue-200 rounded-lg">
          <div className="flex items-center space-x-2">
            <MapPin className="h-4 w-4 text-blue-600" />
            <span className="text-sm text-blue-800">
              Found {evidence.length} evidence source{evidence.length !== 1 ? 's' : ''} for "{query}"
            </span>
          </div>
          <div className="mt-2 text-xs text-blue-700">
            Average relevance score: {(evidence.reduce((sum, item) => sum + item.score, 0) / evidence.length).toFixed(3)}
          </div>
        </div>
      )}
    </div>
  )
}

export default EvidenceBlock
