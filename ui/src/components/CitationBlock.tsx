import React, { useState } from 'react'
import { ChevronDown, ChevronRight, ExternalLink, FileText, MapPin } from 'lucide-react'

interface Citation {
  id: string
  document_id: string
  document_title?: string
  page_number?: number
  section_title?: string
  section_level?: number
  chunk_id: string
  score: number
  content: string
  metadata?: Record<string, any>
}

interface CitationBlockProps {
  citations: Citation[]
  answer: string
}

const CitationBlock: React.FC<CitationBlockProps> = ({ citations, answer }) => {
  const [expandedCitations, setExpandedCitations] = useState<Set<string>>(new Set())
  const [showAllCitations, setShowAllCitations] = useState(false)

  const toggleCitation = (citationId: string) => {
    setExpandedCitations(prev => {
      const newSet = new Set(prev)
      if (newSet.has(citationId)) {
        newSet.delete(citationId)
      } else {
        newSet.add(citationId)
      }
      return newSet
    })
  }

  const formatLocation = (citation: Citation) => {
    const parts = []
    if (citation.section_title) {
      parts.push(citation.section_title)
    }
    if (citation.page_number) {
      parts.push(`Page ${citation.page_number}`)
    }
    return parts.join(', ') || 'Document'
  }

  const getScoreColor = (score: number) => {
    if (score >= 0.8) return 'text-green-600 bg-green-50'
    if (score >= 0.6) return 'text-yellow-600 bg-yellow-50'
    return 'text-red-600 bg-red-50'
  }

  const getScoreLabel = (score: number) => {
    if (score >= 0.8) return 'High'
    if (score >= 0.6) return 'Medium'
    return 'Low'
  }

  const displayedCitations = showAllCitations ? citations : citations.slice(0, 3)

  return (
    <div className="space-y-4">
      {/* AI Answer with Inline Citations */}
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-6">
        <div className="flex items-center mb-3">
          <h4 className="font-semibold text-blue-900 flex items-center">
            <FileText className="h-5 w-5 mr-2" />
            AI Answer
          </h4>
          <span className="ml-auto text-sm text-blue-700">
            Based on {citations.length} source{citations.length !== 1 ? 's' : ''}
          </span>
        </div>
        <div className="prose prose-blue max-w-none">
          <p className="text-blue-800 whitespace-pre-wrap leading-relaxed">
            {answer}
          </p>
        </div>
      </div>

      {/* Citations */}
      <div className="bg-gray-50 border border-gray-200 rounded-lg p-6">
        <div className="flex items-center justify-between mb-4">
          <h4 className="font-semibold text-gray-900 flex items-center">
            <MapPin className="h-5 w-5 mr-2" />
            Sources & Evidence
          </h4>
          {citations.length > 3 && (
            <button
              onClick={() => setShowAllCitations(!showAllCitations)}
              className="text-sm text-blue-600 hover:text-blue-800 font-medium"
            >
              {showAllCitations ? 'Show Less' : `Show All ${citations.length} Sources`}
            </button>
          )}
        </div>

        <div className="space-y-3">
          {displayedCitations.map((citation, index) => (
            <div key={citation.id} className="bg-white border border-gray-200 rounded-lg overflow-hidden">
              <div 
                className="p-4 cursor-pointer hover:bg-gray-50 transition-colors"
                onClick={() => toggleCitation(citation.id)}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-3">
                    <div className="flex items-center">
                      {expandedCitations.has(citation.id) ? (
                        <ChevronDown className="h-4 w-4 text-gray-500" />
                      ) : (
                        <ChevronRight className="h-4 w-4 text-gray-500" />
                      )}
                    </div>
                    <div>
                      <div className="flex items-center space-x-2">
                        <span className="text-sm font-medium text-gray-900">
                          Source {index + 1}
                        </span>
                        <span className={`px-2 py-1 text-xs font-medium rounded-full ${getScoreColor(citation.score)}`}>
                          {getScoreLabel(citation.score)} Relevance
                        </span>
                        <span className="text-xs text-gray-500">
                          {citation.score.toFixed(3)}
                        </span>
                      </div>
                      <div className="text-sm text-gray-600 mt-1">
                        {citation.document_title || `Document ${citation.document_id}`}
                        {formatLocation(citation) && (
                          <span className="ml-2">• {formatLocation(citation)}</span>
                        )}
                      </div>
                    </div>
                  </div>
                  <button className="p-1 text-gray-400 hover:text-gray-600">
                    <ExternalLink className="h-4 w-4" />
                  </button>
                </div>
              </div>

              {expandedCitations.has(citation.id) && (
                <div className="border-t border-gray-200 p-4 bg-gray-50">
                  <div className="space-y-3">
                    <div>
                      <h6 className="text-sm font-medium text-gray-900 mb-2">Content</h6>
                      <p className="text-sm text-gray-700 leading-relaxed">
                        {citation.content}
                      </p>
                    </div>
                    
                    {citation.metadata && Object.keys(citation.metadata).length > 0 && (
                      <div>
                        <h6 className="text-sm font-medium text-gray-900 mb-2">Metadata</h6>
                        <div className="bg-white border border-gray-200 rounded p-3">
                          <dl className="grid grid-cols-1 gap-2 sm:grid-cols-2">
                            {Object.entries(citation.metadata).map(([key, value]) => (
                              <div key={key} className="flex">
                                <dt className="text-xs font-medium text-gray-500 capitalize w-20 flex-shrink-0">
                                  {key.replace(/_/g, ' ')}:
                                </dt>
                                <dd className="text-xs text-gray-700 flex-1">
                                  {typeof value === 'object' ? JSON.stringify(value) : String(value)}
                                </dd>
                              </div>
                            ))}
                          </dl>
                        </div>
                      </div>
                    )}

                    <div className="flex items-center justify-between pt-2 border-t border-gray-200">
                      <div className="flex items-center space-x-4 text-xs text-gray-500">
                        <span>Chunk ID: {citation.chunk_id}</span>
                        <span>Document ID: {citation.document_id}</span>
                      </div>
                      <div className="flex items-center space-x-2">
                        <span className="text-xs text-gray-500">Relevance Score:</span>
                        <span className={`px-2 py-1 text-xs font-medium rounded ${getScoreColor(citation.score)}`}>
                          {citation.score.toFixed(3)}
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>

        {citations.length > 3 && !showAllCitations && (
          <div className="mt-4 pt-4 border-t border-gray-200 text-center">
            <button
              onClick={() => setShowAllCitations(true)}
              className="text-sm text-blue-600 hover:text-blue-800 font-medium"
            >
              View {citations.length - 3} more sources
            </button>
          </div>
        )}
      </div>
    </div>
  )
}

export default CitationBlock
