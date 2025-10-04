import { useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { 
  BookOpen, 
  Download, 
  RefreshCw,
  ChevronRight,
  ChevronDown,
  ExternalLink,
  Info,
  Code,
  Database,
  Brain,
  Search,
  BarChart3,
  Image
} from 'lucide-react'
import toast from 'react-hot-toast'
import { documentationService, DocumentationFile } from '../services/documentationService'

interface LocalDocumentationFile extends DocumentationFile {
  id: string
  icon: any
}

const Methodology = () => {
  const [selectedDoc, setSelectedDoc] = useState<string | null>(null)
  const [markdownContent, setMarkdownContent] = useState<string>('')
  const [loading, setLoading] = useState(false)
  const [expandedSections, setExpandedSections] = useState<Set<string>>(new Set(['overview']))

  const documentationFiles: LocalDocumentationFile[] = [
    {
      id: 'multimodal-processing',
      title: 'Multimodal Processing Guide',
      description: 'Complete guide to table and chart extraction from PDFs',
      filename: 'MULTIMODAL_PROCESSING_GUIDE.md',
      category: 'Processing',
      icon: BarChart3
    },
    {
      id: 'migration-guide',
      title: 'Migration Guide',
      description: 'How to migrate from previous versions',
      filename: 'MIGRATION_GUIDE.md',
      category: 'Setup',
      icon: RefreshCw
    },
    {
      id: 'ocr-replacement',
      title: 'OCR Replacement Guide',
      description: 'Upgrading OCR capabilities and configuration',
      filename: 'OCR_REPLACEMENT_GUIDE.md',
      category: 'Configuration',
      icon: Image
    },
    {
      id: 'rag-upgrade',
      title: 'RAG Upgrade Plan',
      description: 'Retrieval-Augmented Generation system improvements',
      filename: 'RAG_UPGRADE_PLAN.md',
      category: 'Architecture',
      icon: Brain
    },
    {
      id: 'rollback-procedures',
      title: 'Rollback Procedures',
      description: 'How to rollback changes if needed',
      filename: 'ROLLBACK_PROCEDURES.md',
      category: 'Operations',
      icon: RefreshCw
    }
  ]

  const categories = [
    {
      id: 'overview',
      name: 'System Overview',
      description: 'Understanding how the RAG system works',
      icon: Info
    },
    {
      id: 'processing',
      name: 'Document Processing',
      description: 'How documents are processed and indexed',
      icon: Database
    },
    {
      id: 'retrieval',
      name: 'Search & Retrieval',
      description: 'How the system finds relevant information',
      icon: Search
    },
    {
      id: 'architecture',
      name: 'System Architecture',
      description: 'Technical architecture and components',
      icon: Code
    }
  ]

  const loadDocumentation = async (filename: string) => {
    setLoading(true)
    try {
      const content = await documentationService.getDocumentation(filename)
      setMarkdownContent(content)
      setSelectedDoc(filename)
    } catch (error: any) {
      toast.error(`Error loading documentation: ${error.message}`)
      console.error('Error loading documentation:', error)
    } finally {
      setLoading(false)
    }
  }

  const toggleSection = (sectionId: string) => {
    const newExpanded = new Set(expandedSections)
    if (newExpanded.has(sectionId)) {
      newExpanded.delete(sectionId)
    } else {
      newExpanded.add(sectionId)
    }
    setExpandedSections(newExpanded)
  }

  // Custom components for ReactMarkdown to match our design system
  const markdownComponents = {
    h1: ({ children }: any) => (
      <h1 className="text-3xl font-bold text-secondary-900 mb-6 border-b border-secondary-200 pb-2">
        {children}
      </h1>
    ),
    h2: ({ children }: any) => (
      <h2 className="text-2xl font-semibold text-secondary-800 mb-4 mt-8 border-b border-secondary-100 pb-2">
        {children}
      </h2>
    ),
    h3: ({ children }: any) => (
      <h3 className="text-xl font-medium text-secondary-700 mb-3 mt-6">
        {children}
      </h3>
    ),
    h4: ({ children }: any) => (
      <h4 className="text-lg font-medium text-secondary-700 mb-2 mt-4">
        {children}
      </h4>
    ),
    p: ({ children }: any) => (
      <p className="mb-4 text-secondary-700 leading-relaxed">
        {children}
      </p>
    ),
    strong: ({ children }: any) => (
      <strong className="font-semibold text-secondary-900">
        {children}
      </strong>
    ),
    em: ({ children }: any) => (
      <em className="italic text-secondary-700">
        {children}
      </em>
    ),
    code: ({ children, className }: any) => {
      const isInline = !className
      if (isInline) {
        return (
          <code className="bg-gray-100 px-2 py-1 rounded text-sm font-mono text-secondary-800">
            {children}
          </code>
        )
      }
      return (
        <code className="bg-gray-100 px-2 py-1 rounded text-sm font-mono text-secondary-800">
          {children}
        </code>
      )
    },
    pre: ({ children }: any) => (
      <pre className="bg-gray-100 p-4 rounded-lg overflow-x-auto text-sm font-mono mb-4">
        {children}
      </pre>
    ),
    ul: ({ children }: any) => (
      <ul className="list-disc list-inside mb-4 space-y-1 text-secondary-700">
        {children}
      </ul>
    ),
    ol: ({ children }: any) => (
      <ol className="list-decimal list-inside mb-4 space-y-1 text-secondary-700">
        {children}
      </ol>
    ),
    li: ({ children }: any) => (
      <li className="ml-4 mb-1">
        {children}
      </li>
    ),
    blockquote: ({ children }: any) => (
      <blockquote className="border-l-4 border-primary-500 pl-4 italic text-secondary-600 mb-4">
        {children}
      </blockquote>
    ),
    table: ({ children }: any) => (
      <div className="overflow-x-auto mb-4">
        <table className="min-w-full border border-secondary-200 rounded-lg">
          {children}
        </table>
      </div>
    ),
    thead: ({ children }: any) => (
      <thead className="bg-secondary-50">
        {children}
      </thead>
    ),
    tbody: ({ children }: any) => (
      <tbody className="divide-y divide-secondary-200">
        {children}
      </tbody>
    ),
    tr: ({ children }: any) => (
      <tr>
        {children}
      </tr>
    ),
    th: ({ children }: any) => (
      <th className="px-4 py-2 text-left text-sm font-medium text-secondary-900 border-b border-secondary-200">
        {children}
      </th>
    ),
    td: ({ children }: any) => (
      <td className="px-4 py-2 text-sm text-secondary-700 border-b border-secondary-100">
        {children}
      </td>
    ),
    a: ({ href, children }: any) => (
      <a 
        href={href} 
        className="text-primary-600 hover:text-primary-700 underline"
        target="_blank"
        rel="noopener noreferrer"
      >
        {children}
      </a>
    )
  }

  const getDocsByCategory = (category: string) => {
    return documentationFiles.filter(doc => doc.category.toLowerCase() === category.toLowerCase())
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-secondary-900">System Methodology</h1>
        <p className="mt-2 text-secondary-600">
          Documentation and guides for understanding how the RAG system works
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Documentation Navigation */}
        <div className="lg:col-span-1">
          <div className="card p-6">
            <h2 className="text-lg font-semibold text-secondary-900 mb-4 flex items-center">
              <BookOpen className="h-5 w-5 mr-2 text-primary-600" />
              Documentation
            </h2>
            
            <div className="space-y-4">
              {categories.map((category) => {
                const Icon = category.icon
                const docs = getDocsByCategory(category.id)
                const isExpanded = expandedSections.has(category.id)
                
                return (
                  <div key={category.id} className="border border-secondary-200 rounded-lg">
                    <button
                      onClick={() => toggleSection(category.id)}
                      className="w-full px-4 py-3 text-left hover:bg-secondary-50 transition-colors rounded-lg"
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex items-center">
                          <Icon className="h-4 w-4 mr-2 text-primary-600" />
                          <span className="font-medium text-secondary-900">{category.name}</span>
                        </div>
                        {isExpanded ? (
                          <ChevronDown className="h-4 w-4 text-secondary-500" />
                        ) : (
                          <ChevronRight className="h-4 w-4 text-secondary-500" />
                        )}
                      </div>
                      <p className="text-sm text-secondary-600 mt-1 ml-6">{category.description}</p>
                    </button>
                    
                    {isExpanded && docs.length > 0 && (
                      <div className="px-4 pb-3 space-y-2">
                        {docs.map((doc) => {
                          const DocIcon = doc.icon
                          const isSelected = selectedDoc === doc.filename
                          
                          return (
                            <button
                              key={doc.id}
                              onClick={() => loadDocumentation(doc.filename)}
                              className={`w-full text-left p-3 rounded-lg transition-colors ${
                                isSelected 
                                  ? 'bg-primary-50 border border-primary-200 text-primary-900' 
                                  : 'hover:bg-secondary-50 text-secondary-700'
                              }`}
                            >
                              <div className="flex items-center">
                                <DocIcon className="h-4 w-4 mr-2" />
                                <div>
                                  <div className="font-medium text-sm">{doc.title}</div>
                                  <div className="text-xs text-secondary-500 mt-1">{doc.description}</div>
                                </div>
                              </div>
                            </button>
                          )
                        })}
                      </div>
                    )}
                  </div>
                )
              })}
            </div>
          </div>

          {/* Quick Actions */}
          <div className="card p-6 mt-6">
            <h3 className="text-lg font-semibold text-secondary-900 mb-4">Quick Actions</h3>
            <div className="space-y-3">
              <button
                onClick={() => window.open('/api/v1/docs/', '_blank')}
                className="w-full flex items-center px-4 py-2 text-sm text-secondary-700 hover:bg-secondary-50 rounded-lg transition-colors"
              >
                <ExternalLink className="h-4 w-4 mr-2" />
                View All Documentation
              </button>
              <button
                onClick={() => documentationService.downloadDocumentation('MULTIMODAL_PROCESSING_GUIDE.md')}
                className="w-full flex items-center px-4 py-2 text-sm text-secondary-700 hover:bg-secondary-50 rounded-lg transition-colors"
              >
                <Download className="h-4 w-4 mr-2" />
                Download Guide
              </button>
            </div>
          </div>
        </div>

        {/* Content Area */}
        <div className="lg:col-span-2">
          <div className="card p-6">
            {!selectedDoc ? (
              <div className="text-center py-12">
                <BookOpen className="h-16 w-16 mx-auto text-secondary-400 mb-4" />
                <h3 className="text-lg font-medium text-secondary-900 mb-2">
                  Select Documentation to View
                </h3>
                <p className="text-secondary-600">
                  Choose a documentation file from the sidebar to view its contents
                </p>
              </div>
            ) : (
              <div>
                <div className="flex items-center justify-between mb-6">
                  <div>
                    <h2 className="text-2xl font-bold text-secondary-900">
                      {documentationFiles.find(doc => doc.filename === selectedDoc)?.title || selectedDoc}
                    </h2>
                    <p className="text-secondary-600 mt-1">
                      {documentationFiles.find(doc => doc.filename === selectedDoc)?.description}
                    </p>
                  </div>
                  <button
                    onClick={() => window.open(`/api/v1/docs/${selectedDoc}`, '_blank')}
                    className="btn btn-outline btn-sm"
                  >
                    <ExternalLink className="h-4 w-4 mr-2" />
                    Open Raw
                  </button>
                </div>

                {loading ? (
                  <div className="flex items-center justify-center py-12">
                    <RefreshCw className="h-8 w-8 animate-spin text-primary-600 mr-3" />
                    <span className="text-secondary-600">Loading documentation...</span>
                  </div>
                ) : (
                  <div className="prose prose-lg max-w-none">
                    <ReactMarkdown 
                      remarkPlugins={[remarkGfm]}
                      components={markdownComponents}
                    >
                      {markdownContent}
                    </ReactMarkdown>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

export default Methodology
